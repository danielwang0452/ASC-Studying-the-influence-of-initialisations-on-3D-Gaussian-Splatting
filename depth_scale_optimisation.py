import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
from test_depth_anything_v2 import get_depth_maps2
from sh_utils_copy import eval_sh, SH2RGB, RGB2SH
from PyTorch3D_PCD import evaluation
import matplotlib.pyplot as plt
import os
import cv2
import torchvision
from torch.optim.lr_scheduler import  ExponentialLR
import shutil
from pytorch3d.ops import sample_farthest_points

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype = torch.float32

class depth_scale_optimizer(nn.Module):
    '''
    a stack of Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                image=gt_image, gt_alpha_mask=loaded_mask,
                depth_map,
                image_name=cam_info.image_name, uid=id, data_device=args.data_device)
                self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device)
                self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device)
                self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                self.camera_center = self.world_view_transform.inverse()[3, :3]
            the Camera class is defined in cameras.py
    '''
    def __init__(self, cameras,  k=5, start_scale=10.0):
        super().__init__()
        self.cameras = cameras
        self.get_depth_maps(cameras)
        self.n_cameras = len(cameras)
        self.k = k-1
        self.C, self.H, self.W = self.cameras[0].original_image.shape
        self.depth_scales = nn.Parameter(start_scale*torch.ones(self.n_cameras), requires_grad=True).to(dtype)

        #self.get_sfm_scales(cameras)
        #print(self.cameras[0].depth_map.shape, self.cameras[0].sfm_depth_map)

    def prepare_camera(self, tgt_cam_idx, test_cam=False):
        ''' cameras_lists is
        a list [(tgt_camera1, [src_cam1, src_cam2 ... ]),
                (tgt_camera2, [src_cam1, src_cam2 ... ]),
                 ...
                (tgt_cameraN, [src_cam1, src_cam2 ... ])]
        '''
        cams = self.cameras_lists[tgt_cam_idx]
        tgt_cam = cams[0]
        all_src_cam_list = cams[1][:-1]
        #src_cam_list = []
        #for idx in torch.randint(0, self.k+self.j, (self.k,)):
        #    src_cam_list.append(all_src_cam_list[idx])
        if test_cam:
            self.k2 = 1
            src_cam_list = [cams[1][-1]]
        else:
            self.k2 = self.k
            src_cam_list = cams[1][:-1]
        # tgt_camera1, [src_cam1, src_cam2 ... (indices)]
        # define matrices
        tgt_points = torch.empty((self.k2, 3, self.H * self.W))  # shape (C, K, 3, H*W)
        tgt_intrinsics = torch.empty((self.k2, 3, 3))  # shape (C, K, 3, 3)
        cam_transforms = torch.empty((self.k2, 4, 4))
        depth_maps = torch.empty((self.k2, 1, self.H * self.W))  # shape (C, K, 1, H*W)
        src_intrinsics = torch.empty((self.k2, 3, 4))
        src_imgs = torch.empty((self.k2, 3, self.H, self.W))  # shape (C, K, 3, H, W)
        tgt_imgs = torch.empty((self.k2, 3, self.H, self.W))
        # build matrices for given camera
        tgt_cam_points = self.get_points(tgt_cam)
        tgt_intrinsic = self.get_tgt_intrinsic(tgt_cam)
        depth_map = 1 / torch.clamp_min(torch.tensor(tgt_cam.depth_map, dtype=dtype), min=0.01).reshape(
            (self.H * self.W))
        tgt_img = self.get_src_img(tgt_cam)
        for j, src_cam_index in enumerate(src_cam_list):  # iterate over src cams
            src_cam = self.cameras[src_cam_index]
            cam_transform = self.get_cam_transform(tgt_cam, src_cam)
            src_intrinsic = self.get_src_intrinsic(src_cam)
            src_img = self.get_src_img(src_cam)
            # insert matrices at index [j]
            tgt_points[j] = tgt_cam_points
            tgt_intrinsics[j] = tgt_intrinsic
            cam_transforms[j] = cam_transform
            depth_maps[j] = depth_map
            src_intrinsics[j] = src_intrinsic
            src_imgs[j] = src_img
            tgt_imgs[j] = tgt_img
            #print(f'src intrinsic:{src_intrinsic}')
            #print(f'tgt_intrinsic: {tgt_intrinsic}')
            #print(f'cam_transform: {cam_transform}')
        # device
        tgt_points = tgt_points.to(device)
        tgt_intrinsics = tgt_intrinsics.to(device)
        cam_transforms = cam_transforms.to(device)
        self.depth_maps = depth_maps.to(device)
        src_intrinsics = src_intrinsics.to(device)
        self.src_imgs = src_imgs.to(device) # self variables are needed for forward comp
        self.tgt_imgs = tgt_imgs.to(device)
        # precompute left, right
        self.left = torch.matmul(src_intrinsics, cam_transforms)
        self.right = torch.matmul(tgt_intrinsics, tgt_points)

    def loss(self, tgt_cam_idx, original_loss=True, test_cam=False, test_depth_scales=False):
        if test_depth_scales:
            scaled_depth_map = self.test_depth_scales.to(device)[tgt_cam_idx, None, None] * self.depth_maps
            #print(self.test_depth_scales)
        else:
            scaled_depth_map = self.depth_scales.to(device)[tgt_cam_idx, None, None]*self.depth_maps
        #scaled_depth_map = 0.9*torch.ones_like(scaled_depth_map)
        #scaled_depth_map = torch.cat((torch.ones(self.k, 2, self.H*self.W).to(device), scaled_depth_map), dim=1)
        tgt_cam_homo_coords = torch.cat((scaled_depth_map * self.right,
                                         torch.ones(self.k2, 1, self.H * self.W).to(device)), dim=1)
        src_img_coords = torch.matmul(self.left, tgt_cam_homo_coords)
        # normalize z coordinate
        X = src_img_coords[:, 0]
        Y = src_img_coords[:, 1]
        Z = src_img_coords[:, 2].clamp(min=1e-3)

        X_norm = 2 * (X / Z) / (
                    self.W - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2 * (Y / Z) / (self.H - 1) - 1  # Idem [B, H*W]
        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        pixel_coords = pixel_coords.reshape((self.k2, self.H, self.W, 2))

        # shape (CK, 3, HW) -> (CK, H, W, 2)
        # bilinear interpolation
        projected_imgs = F.grid_sample(self.src_imgs, pixel_coords, padding_mode='zeros', align_corners=True)
        # shape (CK, 3, H, W)
        valid_points = pixel_coords.abs().max(dim=-1)[0] <= 1
        if original_loss:
            loss = ((self.tgt_imgs - projected_imgs) * (valid_points).unsqueeze(
                1)).float().abs().mean()
        if test_cam==True:
            loss = ((self.tgt_imgs - projected_imgs) * (valid_points).unsqueeze(
                1)).float().abs()
            return loss
        else:
            loss = ((self.tgt_imgs - projected_imgs) * (valid_points).unsqueeze(
            1)).float().abs().sum()/(projected_imgs.sum()+1)
            #print(projected_imgs.sum())
        valid = valid_points.sum()/(self.H*self.W*self.k2)
        #loss = valid_points.long().sum()
        return loss, projected_imgs, valid

    def select_points(self, cam_idx, k=1000):
        i = cam_idx
        # warp scaled depth map to test view, sample points with lowest l2 loss
        #self.test_depth_scales = 0.01*torch.ones_like(self.depth_scales)
        cam_list = self.cameras_lists[i]
        tgt_cam = cam_list[0]
        src_cam = cam_list[1][-1]
        self.prepare_camera(i, test_cam=True)
        loss = self.loss(i, test_cam=True, test_depth_scales=True).detach().cpu().squeeze()
        loss = loss.sum(dim=0)
        loss[loss == 0] = 10.0
        values, indices_1d = torch.topk(loss.flatten(), k, largest=False)
        # print(torch.unravel_index(indices_1d, loss.shape))
        # loss is shape H, W
        indices_2d = torch.column_stack(torch.unravel_index(indices_1d, loss.shape))
        masked_loss = loss[indices_2d[:, 0], indices_2d[:, 1]]
        return indices_2d

    def unproject_points(self, pts_per_cam=1000, downsample_scale=1):
        #all_points = torch.empty(len(self.cameras)*pts_per_cam, 3)
        #all_colours = torch.empty(len(self.cameras)*pts_per_cam, 3)
        all_points = None
        all_colours = None
        y_indices, x_indices = np.indices((int(self.H/downsample_scale), int(self.W/downsample_scale)))
        img_coords = torch.cat((torch.tensor(x_indices, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(y_indices, dtype=torch.float32).unsqueeze(0),
                                ), dim=0).to(device) #(2, H, W)
        '''
        self.test_depth_scales = torch.tensor([12.6812, 15.9864, 10.8540, 11.1602, 13.4659, 11.2757, 14.1467, 10.0754,
        '''
        for i, cam_list in enumerate(self.cameras_lists):
            # downsample image colour, depth map
            downsample_fn = torchvision.transforms.Resize(size=(int(self.H / downsample_scale),
                                                                int(self.W / downsample_scale)))
            tgt_cam = cam_list[0]
            image = downsample_fn.forward(torch.tensor(tgt_cam.original_image))
            depth_map = downsample_fn.forward(1 / torch.clamp_min(
                torch.tensor(tgt_cam.depth_map, dtype=dtype).unsqueeze(0), min=0.4
                )).squeeze()
            multiplier = 1
            selected = 0
            while selected < pts_per_cam:
                indices_2d = self.select_points(i, k=int(pts_per_cam*multiplier))
                indices_removed = torch.unique(torch.round((indices_2d) / downsample_scale), dim=0).to(torch.int)
                x = indices_removed[:, 0]
                y = indices_removed[:, 1]
                print(x.max(), y.max())
                x[x == x.max()] = x.max() - 1
                y[y == y.max()] = y.max() - 1
                print(x.max(), y.max())
                indices_removed = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=1)
                multiplier *= 1.5
                selected = indices_removed.shape[0]
            indices_removed = indices_removed[:pts_per_cam]
            print(f'adasdad:{x.max(), y.max()}')
            selected_points = torch.cat((
                img_coords[:, indices_removed[:, 0], indices_removed[:, 1]],
                torch.ones((1, indices_removed.shape[0])).to(device)), dim=0)
            #print(indices_2d.shape)
            #indices_removed=torch.round((indices_2d-1)/downsample_scale).to(torch.int)

            #print(indices_2d[:10])
            print(selected_points.shape)
            print(indices_removed.shape)
            #indices_removed=indices_2d
            #print(indices_removed)
            #print(indices_removed.shape)
            selected_colours = image[:, indices_removed[:, 0], indices_removed[:, 1]]
            intrinsic = self.get_tgt_intrinsic(tgt_cam).to(device).inverse()
            intrinsic[:2, :] = intrinsic[:2, :] / downsample_scale
            intrinsic = intrinsic.inverse()
            extrinsic = torch.tensor(tgt_cam.world_view_transform).transpose(0, 1).inverse()
            depth_map = depth_map[indices_removed[:, 0], indices_removed[:, 1]].to(device)
            scaled_depth = self.test_depth_scales[i]*depth_map[None, :]
            cam_coords = intrinsic @ selected_points # (3, 1000)
            cam_coords = torch.cat((scaled_depth*cam_coords,torch.ones((1, indices_removed.shape[0])).to(device)), dim=0)
            world_coords = extrinsic @ cam_coords
            #all_points[i*pts_per_cam:(i+1)*pts_per_cam, :] = world_coords[:3].T
            #all_colours[i*pts_per_cam:(i+1)*pts_per_cam, :] = selected_colours.T
            if all_points == None:
                all_points = world_coords[:3].T
                all_colours = selected_colours.T
            else:
                all_points = torch.cat((all_points, world_coords[:3].T), dim=0)
                all_colours = torch.cat((all_colours, selected_colours.T), dim=0)
            print(all_points.shape)
        return all_points, RGB2SH(all_colours)

    def render_pcd_depth(self, sfm_pcd, take_nearest=False):
        scales = []
        for t, tgt_cam in enumerate(self.cameras):
            #all_src_cam_list = cams[1]
            d_scale = self.test_depth_scales[t]
            #d_scale = 12.0*torch.ones_like(self.test_depth_scales[t])
            extrinsic = tgt_cam.world_view_transform
            pcd = torch.cat((sfm_pcd, torch.ones(sfm_pcd.shape[0], 1).to(device)), dim=1)
            cam_coords = extrinsic.T @ pcd.permute((1, 0))
            intrinsic = torch.cat((torch.tensor(tgt_cam.intrinsic_matrix, dtype=dtype),
                              torch.zeros(3, 1)), dim=1).to(device)
            # mask points behind camera
            z_mask = (cam_coords[2]>0).to(torch.int32)
            z_indices = torch.nonzero(z_mask)
            pos_cam_coords = torch.cat((
                cam_coords[0][z_indices].T,
                cam_coords[1][z_indices].T,
                cam_coords[2][z_indices].T,
                cam_coords[3][z_indices].T
            ), dim=0)
            img_coords = intrinsic @ pos_cam_coords
            # normalize z coordinate
            X = img_coords[0]
            Y = img_coords[1]
            Z = img_coords[2].clamp(min=1e-3)

            X_norm = 2 * (X / Z) / (
                   self.W - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
            Y_norm = 2 * (Y / Z) / (self.H - 1) - 1  # Idem [B, H*W]

            pixel_coords = torch.stack([X_norm, Y_norm], dim=1)  # [B, H*W, 2]
            # extract points with:
            # 2) 0 < x < image width, 3) 0 < y < image height
            mask_2 = (-1 < X_norm)*(X_norm<1)
            mask_3 = (-1 < Y_norm) * (Y_norm <1)
            #print(mask_2[:50])
            mask = (mask_2*mask_3)
            xys = torch.cat(((X/Z).unsqueeze(0), (Y/Z).unsqueeze(0)), dim=0).T
            # shape (2, N)
            monodepthmap = 1 / torch.clamp_min(torch.tensor(tgt_cam.depth_map, dtype=dtype).T,
                                               min=0.1)
            sfm_depth = pos_cam_coords[2, :][mask]
            if take_nearest:
                count=0 # count number of replaced pixels
                # construct grid of same shape as depth map, for each valid sfm point
                # place it in the corresponding pixel if it is the closest point for that pixel
                grid = 10000*torch.ones_like(monodepthmap).to(torch.float32)
                for p, point in enumerate(xys[mask, :]):
                    x, y = int(point[0]), int(point[1])
                    if sfm_depth[p] < grid[x, y]:
                        if grid[x, y] == 10000:
                            count +=1
                        grid[x, y] = sfm_depth[p]
                grid[grid==10000] = 0
                scale = (grid[grid>0].median() / torch.tensor(monodepthmap[grid>0]).median())
                scales.append(scale)
                # save depth map
                disparity = 1 / (grid + 1e-5)
                disparity[disparity==1/(1e-5)] = 0
                max, min = disparity.max(), disparity.min()
                disparity_normalised = (disparity - min) / (max - min)
                torchvision.utils.save_image(disparity_normalised.T,
                    fp=f'sfm_rendered_depths/cam_{t}.png')
                t_dsfm = grid[grid>0].median()
                s_dsfm = (grid[grid>0]-t_dsfm).abs().mean()
                t_d = monodepthmap[grid>0].median()
                s_d = (monodepthmap[grid>0]-t_d).abs().mean()
                scale2 = s_dsfm/s_d

                difference = d_scale.cpu()*torch.tensor(monodepthmap[grid>0])-grid[grid>0]
                print(d_scale, difference.mean())
                #print(' ')
                #print(scale)
                #print(scale2)

            else:
                iterations = int(xys[mask, :].shape[0]/32766) + 1
                monodepths = None
                #print(xys.shape)
                for i in range(iterations):
                    if i == iterations-1: # last iter
                        maps = xys[mask, :][i*32766:].cpu().detach().numpy()
                    else:
                        maps = xys[mask, :][i*32766:(i+1)*32766].cpu().detach().numpy()
                    monodepth = cv2.remap(monodepthmap.numpy(), maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)[..., 0]
                    if monodepths is None:
                        monodepths = monodepth
                    else: monodepths = np.concatenate((monodepths, monodepth))
                #print(monodepths.shape)
                scale = (sfm_depth.median()/torch.tensor(monodepth).median())
                scales.append(scale)
                # save depth map
        self.sfm_scales = torch.tensor(scales, dtype=dtype)
        # pcd is shape N, 3

    def get_k_nearest_cameras(self, k_cams):
        '''
        Returns a list [(tgt_camera1, [src_cam1, src_cam2 ... ]),
                         (tgt_camera2, [src_cam1, src_cam2 ... ]),
                         ...
                         (tgt_cameraN, [src_cam1, src_cam2 ... ])]
        '''
        camera_centres = torch.empty((self.n_cameras, 3))
        camera_rots = torch.empty((self.n_cameras, 9))
        for i, camera in enumerate(self.cameras):
            camera_centres[i, :] = torch.tensor(camera.camera_center, dtype=dtype)
            camera_rots[i, :] = torch.tensor(camera.R, dtype=dtype).flatten()
        # centres
        camera_centres = camera_centres.unsqueeze(0).repeat((self.n_cameras, 1, 1))
        camera_centres_t = camera_centres.permute((1, 0, 2))
        distances = torch.square(camera_centres - camera_centres_t).sum(dim=2) # (N, N)
        # rotations
        camera_rots = camera_rots.unsqueeze(0).repeat((self.n_cameras, 1, 1))
        camera_rots_t = camera_rots.permute((1, 0, 2))
        distances2 = torch.square(camera_rots - camera_rots_t).sum(dim=2)
        #
        indices2 = torch.topk(
            torch.nn.functional.normalize(distances) + torch.nn.functional.normalize(distances2),
            k_cams + 1, dim=1, largest=False, sorted=True
        )[1]
        indices = torch.topk(1*(distances-distances.mean())/distances.std() +
                             1*(distances2-distances2.mean())/distances2.std(), k_cams+1, dim=1, largest=False, sorted=True)[1] # (N, k+1)
        #print(indices, indices2)
        # use indices to construct list of tgt - src cameras
        cameras_list = []
        for n, src_cam_indices in enumerate(indices.tolist()): # iterate over each tgt camera
            # for each n, get (camera_n, [src_cam1, src_cam2 ...])
            # ignore camera distance that corresponds to itself
            cameras_list.append(
                (self.cameras[n], src_cam_indices[1:])
            )
        self.cameras_lists = cameras_list

    def get_points(self, camera):
        y_indices, x_indices = np.indices((self.H, self.W))
        img_coords = torch.cat((torch.tensor(x_indices, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(y_indices, dtype=torch.float32).unsqueeze(0),
                                torch.ones((self.H, self.W), dtype=torch.float32).unsqueeze(0)), dim=0)
        img_coords = img_coords.reshape(self.C, self.H * self.W)  # (3, H*W)
        return img_coords

    def get_tgt_intrinsic(self, camera):
        return torch.tensor(camera.intrinsic_matrix, dtype=dtype).inverse()

    def get_cam_transform(self, tgt_camera, src_camera):
        #print(tgt_camera.world_view_transform)
        return torch.matmul(torch.tensor(src_camera.world_view_transform).transpose(0, 1),
                             torch.tensor(tgt_camera.world_view_transform).transpose(0, 1).inverse())

    def get_depth_maps(self, cameras):
        depth_maps = get_depth_maps2(cameras)
        # initialises every camera.depth_map
        for i, camera in enumerate(cameras):
            H, W = camera.depth_map.shape
            #camera.depth_map = 1/torch.clamp_min(torch.tensor(camera.depth_map, dtype=dtype), min=0.4)
        #return depth_map.reshape((H*W))

    def get_src_intrinsic(self, camera):
        return torch.cat((torch.tensor(camera.intrinsic_matrix),
                          torch.zeros(3, 1)), dim=1)

    def get_src_img(self, camera):
        return torch.tensor(camera.original_image, dtype=dtype)

    # for scene initialisation
    def get_cam_to_world(self, camera):
        # for some reason they store the extrinsic matrix transposed
        return torch.tensor(camera.world_view_transform).transpose(0, 1).inverse()

    def get_colours(self, camera):
        return torch.tensor(camera.original_image).reshape(self.C, self.H*self.W)

    def init_depth_scale(self, tgt_cam_idx,
                               scales=None,
                               n_iterations=1,
                               plot_losses=False):
        losses = torch.ones_like(scales)
        if os.path.exists(f'inverse_warp_imgs/cam{tgt_cam_idx}'):
            shutil.rmtree(f'inverse_warp_imgs/cam{tgt_cam_idx}')
        os.makedirs(f'inverse_warp_imgs/cam{tgt_cam_idx}', exist_ok=True)
        self.prepare_camera(tgt_cam_idx)
        warped_imgs = []
        for s, scale in enumerate(scales):
            self.depth_scales = nn.Parameter(scale*torch.ones(self.n_cameras), requires_grad=False).to(dtype)
            cam_loss, warped_img, valid = self.loss(tgt_cam_idx, original_loss=False)
            if s in [2, 14]:
                print(scale)
                warped_imgs.append(warped_img)
            losses[s] = cam_loss.item()
            #if valid < 0.5:
            #    losses[s] = 1.0
            '''    
            for k in range(self.k):
                src_1 = torch.cat(
                    (self.tgt_imgs[0].cpu().detach().unsqueeze(0), self.src_imgs[k].cpu().detach().unsqueeze(0), warped_img.cpu().detach()[k].unsqueeze(0)), dim=0)
                torchvision.utils.save_image(src_1, fp=f'inverse_warp_imgs/cam{tgt_cam_idx}/tgt_src_warp_{k}_{scale}.png')
                '''
        '''
        for k in range(self.k):
            src_1 = torch.cat(
                (self.tgt_imgs[0].cpu().detach().unsqueeze(0), self.src_imgs[k].cpu().detach().unsqueeze(0),
                 warped_imgs[0].cpu().detach()[k].unsqueeze(0), warped_imgs[1].cpu().detach()[k].unsqueeze(0)), dim=0)
            torchvision.utils.save_image(src_1, fp=f'inverse_warp_imgs/cam{tgt_cam_idx}/tgt_src_warp_{k}_{scale}.png')
        '''
        #print(losses)
        #if torch.argmin(losses) == 0 or torch.argmin(losses) == len(scales):
        # train losses
        plot_losses=False
        if plot_losses:
            #if os.path.exists(f'init_loss_plots'):
            #    shutil.rmtree(f'init_loss_plots')
            #    os.makedirs(f'init_loss_plots', exist_ok=True)
            plt.figure(figsize=(10, 5))
            # plt.plot(scale_means, train_losses, marker='o', linestyle='-', color='blue', label='Chamfer Loss')
            plt.plot(scales, losses, marker='o', linestyle='-', color='blue', label='Chamfer Loss')
            # for i in range(len(train_losses)):
            #    plt.text(i, train_losses[i], f'{scale_means[i]:.2f}', fontsize=9, ha='right', va='bottom')
            plt.title(f'Image warp losses')
            plt.xlabel('Scale')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f'init_loss_plots/{tgt_cam_idx}.png')
            plt.close()
            #
        chosen_scale = scales[torch.argmin(losses)]
        if chosen_scale < 9.0:
            chosen_scale = 15.0
        print(f'chosen scale: {chosen_scale}')
        return chosen_scale

def plot_only_train_losses(train_losses, scale_means, path=f'eval_graphs/train_loss_plot_test.png'):
    # train losses
    plt.figure(figsize=(10, 5))
    #plt.plot(scale_means, train_losses, marker='o', linestyle='-', color='blue', label='Chamfer Loss')
    plt.plot(train_losses, marker='o', linestyle='-', color='blue', label='Chamfer Loss')
    for i in range(len(train_losses)):
        plt.text(i, train_losses[i], f'{scale_means[i]:.2f}', fontsize=9, ha='right', va='bottom')
    plt.title(f'Image warp losses')
    plt.xlabel('Scale')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def save_img_warps(tgt_idx, tgt_cam, src_imgs, tgt_img, warped_imgs, scale):
    def save_image(depth_map, path):
        torchvision.utils.save_image(depth_map, fp=path)
        '''
        depth_map = depth_map.permute((1, 2, 0)).cpu().detach().numpy()
        # first normalize depth map
        depth_normalized = 255 * (depth_map)
        depth_normalized = depth_normalized.astype(np.uint8)
        image = Image.fromarray(depth_normalized)
        # Define the file path
        # Save the image
        image.save(path)
        '''
        return
    i = tgt_idx
    '''
    os.makedirs(f'inverse_warp_imgs/cam{i}', exist_ok=True)
    # save tgt_img
    save_image(tgt_img, f'inverse_warp_imgs/cam{i}/tgt.png')
    # save src imgs
    for k, src_img in enumerate(src_imgs):
        save_image(src_img, f'inverse_warp_imgs/cam{i}/src_img{k}.png')
    # save warped_imgs
    for w, warped_img in enumerate(warped_imgs):
        save_image(warped_img, f'inverse_warp_imgs/cam{i}/warped_img{w}_{scale}.png')
    '''

def optimize_depth_scale(cameras, sfm_xyz, eval,
                         k=3, n_iterations=100, lr=1e-1, start_scale=12.0):
    print(f'(sfm shape: {sfm_xyz.shape}')
    eval=False
    # use sfm_xyz, sfm_colour, sfm_scales for evaluation
    model = depth_scale_optimizer(cameras, k, start_scale).to(device)
    evaluator = evaluation(model.cameras, n_random_pts=1000000, n_farthest_pts=350000)
    evaluator.depth_scales = model.depth_scales.detach()
    # optimize depth scales
    model.get_k_nearest_cameras(k_cams=k)
    all_cam_losses = []
    all_cam_scales = []
    # search for best depth scale for n cameras
    model.eval()

    with torch.no_grad():
        chosen_scales = torch.ones(len(cameras))
        for i, init_cam in enumerate(cameras):
            #chosen_scale = model.init_depth_scale(i, scales=torch.arange(1, 21, dtype=dtype))
            chosen_scale = model.init_depth_scale(i, scales=2*torch.arange(1, 20, dtype=dtype))
            chosen_scales[i] = chosen_scale
            #break
            print(i)
        init_scale = torch.tensor(chosen_scales).mean()
        print(f'init scale: {init_scale}')
        model.depth_scales = nn.Parameter(torch.tensor(chosen_scales), requires_grad=True)
        #model.depth_scales = nn.Parameter(12.0*torch.ones_like(chosen_scales), requires_grad=True)
        print(model.depth_scales)
    # now optimise
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i in range(len(cameras)): # iterate over tgt cams
        cam_losses = []
        cam_scales = []
        #if os.path.exists(f'inverse_warp_imgs/cam{i}'):
        #    shutil.rmtree(f'inverse_warp_imgs/cam{i}')
        model.prepare_camera(i)
        for iteration in range(n_iterations):
            #model.depth_scales = nn.Parameter((3 * iteration+1) * torch.ones_like(model.depth_scales),requires_grad=False)
            cam_loss, projected_imgs, valid = model.loss(i, original_loss=False)

            optimizer.zero_grad()
            cam_loss.backward()
            optimizer.step()

            cam_losses.append(cam_loss.item())
            cam_scales.append(model.depth_scales[i].item())
            evaluator.depth_scales[i] = model.depth_scales[i].detach().to(device)

        print(f'{i}, {model.depth_scales[i]}, {cam_loss.item()}')
        all_cam_losses.append(cam_losses)
        all_cam_scales.append(cam_scales)
        cameras[i].depth_scale = model.depth_scales[i]
        #plot_only_train_losses(cam_losses[::3], cam_scales[::3], path=f'inverse_warp_imgs/cam{i}/train_loss.png')
        # save img
        save_img_warps(tgt_idx=i, tgt_cam=cameras[i],src_imgs=model.src_imgs.cpu(),tgt_img=model.tgt_imgs[0].cpu(),warped_imgs=projected_imgs.cpu().detach(), scale=model.depth_scales.mean().cpu())
        #break

    print(f'final scales: {model.depth_scales.mean()}')
    all_cam_losses = torch.tensor(all_cam_losses)
    all_cam_scales = torch.tensor(all_cam_scales)
    if eval: # compute chamfer distances
        with torch.no_grad():
            scales = 3*torch.arange(10)+1.0
            evaluator.scale_means = scales
            all_scale_losses = [] # should be shape (n_scales, n_cams, 1 iter)
            for scale in scales:
                scale_losses = []
                evaluator.depth_scales = scale*torch.ones_like(model.depth_scales).to(torch.float32)
                model.depth_scales = nn.Parameter(evaluator.depth_scales)
                evaluator.sfm_eval_loss(cameras, sfm_xyz)
                # get img warp losses
                #for n in range(len(cameras)):  # iterate over tgt cams
                #    model.prepare_camera(n)
                #    cam_loss = model.loss(n)
                #    scale_losses.append(cam_loss.item())
                #all_scale_losses.append(scale_losses)
        #evaluator.train_losses = torch.tensor(all_scale_losses).mean(dim=1).numpy()
        evaluator.plot_losses()
    with torch.no_grad():
        evaluator.depth_scales = model.depth_scales.detach().to(device)
        #evaluator.depth_scales = model.sfm_scales.detach().to(device)
        #evaluator.depth_scales = 12.0*torch.ones_like(model.depth_scales.detach()).to(device)
        #model.test_depth_scales = torch.
        model.test_depth_scales = model.depth_scales


        #model.test_depth_scales = 12.0*torch.ones_like(model.test_depth_scales)

        #model.render_pcd_depth(sfm_xyz, take_nearest=True)
        plot_only_train_losses(train_losses=all_cam_losses.mean(dim=0),
                               scale_means=all_cam_scales.mean(dim=0))
        # manually set scale for pcd
        #evaluator.depth_scales = 12.0 * torch.ones_like(model.depth_scales)
        print(f's: {evaluator.depth_scales}')

        #points, colours, gaussian_scales = evaluator.get_point_cloud(cameras, n_points=35000, return_gaussian_scales=True)
        points, colours = model.unproject_points(pts_per_cam=int(1000000/len(cameras)))

        gaussian_scales = torch.log(torch.clamp_min(0.05 * torch.rand((points.shape[0]))[..., None].repeat(1, 3), 0.0000001)).to(device)
        #opacities = torch.sigmoid((gaussian_scales - gaussian_scales.mean())/gaussian_scales.std()).unsqueeze(-1)
        opacities = 1.0*torch.ones_like(gaussian_scales[:, 0]).unsqueeze(-1)
        print(points.shape)
    return points.cpu(), colours.cpu(), gaussian_scales.cpu(), opacities.cpu()

def test_depth_scale(cameras, sfm_xyz, k=5, n_iterations=100, lr=1e-1):
    print(f'(sfm shape: {sfm_xyz.shape}')
    eval=False
    # use sfm_xyz, sfm_colour, sfm_scales for evaluation
    model = depth_scale_optimizer(cameras, k, start_scale=12.0).to(device)
    evaluator = evaluation(model.cameras, n_random_pts=50000, n_farthest_pts=35000)
    evaluator.depth_scales = model.depth_scales.detach()
    # optimize depth scales
    model.get_k_nearest_cameras()
    all_cam_losses = []
    all_cam_scales = []
    # search for best depth scale for n cameras
    model.eval()
    model.render_pcd_depth(sfm_xyz, take_nearest=True)

    with torch.no_grad():
        chosen_scales = []
        for i, init_cam in enumerate(cameras):
            print(i)
            chosen_scale = model.init_depth_scale(i, scales=2*torch.arange((20), dtype=dtype)
                                                   , plot_losses=True)
            chosen_scales.append(chosen_scale)
            print(f'sfm scale:{model.sfm_scales[i]}')
            #break
        init_scale = torch.tensor(chosen_scales).mean()
        print(f'init scale: {init_scale}')
        model.depth_scales = nn.Parameter(torch.tensor(chosen_scales), requires_grad=True)
        # manual init
        #model.depth_scales = nn.Parameter(14 * torch.ones((len(cameras),)), requires_grad=True)
    # now optimise
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for i in range(len(cameras)): # iterate over tgt cams
        shutil.rmtree(f'inverse_warp_imgs/cam{i}')
        os.makedirs(f'inverse_warp_imgs/cam{i}', exist_ok=True)
        cam_losses = []
        cam_scales = []
        # look at train i+13 (decreasing), i+15 (normal)
        model.prepare_camera(i)

        for iteration in range(n_iterations):
            #model.depth_scales = nn.Parameter((4 * iteration+1) * torch.ones_like(model.depth_scales),requires_grad=False)
            cam_loss, projected_imgs, valid = model.loss(i, original_loss=False)
            #print(model.depth_scales[i])
            optimizer.zero_grad()
            cam_loss.backward()
            optimizer.step()

            cam_losses.append(cam_loss.item())
            cam_scales.append(model.depth_scales[i].item())
            evaluator.depth_scales[i] = model.depth_scales[i].detach().to(device)
            # save img
            save_img_warps(tgt_idx=i, tgt_cam=cameras[i],src_imgs=model.src_imgs.cpu(),tgt_img=model.tgt_imgs[0].cpu(),warped_imgs=projected_imgs.cpu().detach(), scale=model.depth_scales.mean().cpu())
            plot_only_train_losses(cam_losses[::3], cam_scales[::3], path=f'inverse_warp_imgs/cam{i}/train_loss.png')
        break
        print(f'{i}, {model.depth_scales[i]}, {cam_loss.item()}')
        all_cam_losses.append(cam_losses)
        all_cam_scales.append(cam_scales)
        cameras[i].depth_scale = model.depth_scales[i]

    print(f'final scales mean: {model.depth_scales.mean()}')
    all_cam_losses = torch.tensor(all_cam_losses)
    all_cam_scales = torch.tensor(all_cam_scales)
    #plot_only_train_losses(cam_losses, cam_scales)

def test_sfm_scale(cameras, sfm_xyz, sfm_colours):
    model = depth_scale_optimizer(cameras, k=3, start_scale=12.0).to(device)
    for i in range(len(cameras)):
        scale = model.render_pcd_depth(sfm_xyz, i)


