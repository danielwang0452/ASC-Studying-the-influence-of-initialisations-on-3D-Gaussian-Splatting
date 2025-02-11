import numpy as np
import torch
import pytorch3d
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
from pytorch3d.renderer.points.renderer import PointsRenderer
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings
from pytorch3d.renderer import AlphaCompositor
from utils.sh_utils import RGB2SH
import matplotlib.pyplot as plt
from sfm_scale import get_sfm_depth
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#device='cpu'
dtype = torch.float32

class evaluation():
    def __init__(self, GS_cameras, n_random_pts=50000, n_farthest_pts=35000, use_sfm=False):
        C, H, W = GS_cameras[0].original_image.shape
        self.use_sfm = use_sfm
        self.n_random_pts = n_random_pts
        self.n_farthest_pts = n_farthest_pts
        self.depth_scales = torch.ones(len(GS_cameras))
        self.gt_precision = []
        self.gt_accuracy = []
        self.gt_loss = []
        self.sfm_precision = []
        self.sfm_accuracy = []
        self.sfm_loss = []
        self.train_losses = []
        self.scale_means = []
        self.depth_maps = torch.empty(len(GS_cameras), 1, H, W)
        self.image_rgbs = torch.empty(len(GS_cameras), C, H, W)
        if use_sfm:
            self.all_sfm_depths = get_sfm_depth()
        self.pytorch_cameras = self.get_pytorch_cameras(GS_cameras)

    def get_pytorch_cameras(self, GS_cameras):
        N = len(GS_cameras)
        self.n_pts_total = N
        C, H, W = GS_cameras[0].original_image.shape
        image_size = torch.tensor((H, W)).unsqueeze(0).repeat((N, 1))
        focal_length = torch.empty((N, 2))
        principal_point = torch.empty((N, 2))
        R = torch.empty((N, 3, 3))
        T = torch.empty((N, 3))
        for n, camera in enumerate(GS_cameras):
            #print(camera.fx, camera.fy, camera.px, camera.py)
            focal_length[n] = torch.tensor((camera.fx, camera.fy))
            principal_point[n] = torch.tensor((camera.px, camera.py))
            R[n] = torch.tensor((camera.R))
            T[n] = torch.tensor((camera.T))
            self.depth_maps[n] = 1/torch.clamp_min(torch.tensor(camera.depth_map, dtype=dtype),
                                                       min=0.4)

                #self.depth_maps[n] = torch.ones_like(self.depth_maps[n])
            #self.depth_maps[n] = camera.sfm_depth_map/self.depth_scales[n]
            # use min=0.4 for initialisation
            # sfm depth
            self.image_rgbs[n] = torch.tensor(camera.original_image, dtype=dtype)
            if self.use_sfm:
                for sfm_depth in self.all_sfm_depths:
                    if camera.image_name == sfm_depth[0]:
                        self.depth_maps[n] = 1 / torch.clamp_min(torch.tensor(sfm_depth[1], dtype=dtype),
                                                min=0.01)
                        print(( 1/torch.clamp_min(torch.tensor(camera.depth_map, dtype=dtype),
                                                       min=0.4)).mean(), (1 / torch.clamp_min(torch.tensor(sfm_depth[1], dtype=dtype),
                                                min=0.01).mean()))

        self.depth_maps = self.depth_maps.to(device)
        cams = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            image_size=image_size,
            in_ndc=False,
            device=device
        )
        #print(R[:5], T[:5])
        return cams

    def get_point_cloud(self, GS_cameras, n_points=None, return_gaussian_scales=False, n_subcams=70):
        self.depth_scales = self.depth_scales.to(device)
        #self.depth_scales = 12.5*torch.ones_like(self.depth_scales)
        #if self.use_sfm:
            #self.depth_maps = torch.ones_like(self.depth_maps)

        N = len(GS_cameras)
        C, H, W = GS_cameras[0].original_image.shape
        '''
        pcd = get_rgbd_point_cloud(
            camera=pytorch_cameras.to(device),
            image_rgb=image_rgbs.to(device),
            depth_map=depth_maps.to(device),
            mask=masks,
            euclidean=False)  # false -> perp distance; true -> exact distance
        '''
        # subsample cameras
        if n_subcams > N:
            n_subcams = N
        indices = torch.randperm(N)[:n_subcams]
        y_indices, x_indices = np.indices((H, W))

        depth_maps = self.depth_scales[indices, None, None, None]*self.depth_maps[indices]
        img_coords = torch.cat((torch.tensor(x_indices, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(y_indices, dtype=torch.float32).unsqueeze(0),
                                ), dim=0).to(device)
        img_coords = torch.cat((img_coords.unsqueeze(0).repeat((n_subcams, 1, 1, 1)), depth_maps.to(device)), dim=1).to(device)
        xy_depth = img_coords.reshape(n_subcams, C, H * W).permute((0, 2, 1))#.reshape(N*H*W, C).to(device)  # (3, H*W)
        points3d = self.pytorch_cameras[indices].unproject_points(xy_depth).reshape(n_subcams*H*W, C)
        gaussian_scales = self.depth_maps[indices].flatten().to(device) # shape N*H*W
        colours = self.image_rgbs[indices].reshape(n_subcams, C, H * W).permute((0, 2, 1)).to(device)
        subsampled_points, subsampled_colours, subsampled_gaussian_scales = self.subsample_points(points3d, colours, n_points=n_points, gs_scales=gaussian_scales)
        # end strategic sampling
        subsampled_colours = RGB2SH(subsampled_colours)
        #subsampled_colours *= 255.0
        if return_gaussian_scales == True:
            return subsampled_points.squeeze(), subsampled_colours.squeeze(), subsampled_gaussian_scales.squeeze()
        return subsampled_points.squeeze(), subsampled_colours.squeeze()

    def subsample_points(self, points, colours=None, n_points=None, gs_scales=None):
        if n_points == None:
            n_points = self.n_farthest_pts
        N, C = points.shape
        self.n_pts_total = N
        if N > self.n_random_pts:
            subsample_idx = torch.randperm(N, device=device)[:self.n_random_pts]
            points = points[subsample_idx]
            if colours != None:
                colours = colours.reshape(N, C)[subsample_idx]
            if gs_scales != None:
                gs_scales = gs_scales[subsample_idx]
            return points, colours, gs_scales
            '''
            # farthest dist subsampling
            print('begin subsampling')
            points, indices = sample_farthest_points(
                points=points.unsqueeze(0).cpu(),
                lengths=torch.tensor(self.n_random_pts).unsqueeze(0).cpu(),
                K=n_points,
                random_start_point=True)
            print('end subsampling')
            if colours != None:
                colours = colours[indices]
                if gs_scales != None:
                    gs_scales = gs_scales[indices]
                    return points, colours, gs_scales
                return points, colours
        return points
        '''

    # compute chamfer loss between current dept scale pcd and sfm initialised pcd
    def sfm_eval_loss(self, gs_cameras, sfm_xyz, gt_xyz):
        subsample_idx = torch.randperm(self.n_pts_total)[:self.n_farthest_pts]
        #sfm_xyz = sfm_xyz[subsample_idx]
        points, colours = self.get_point_cloud(gs_cameras)
        # eval SfM pcd
        sfm_accuracy = chamfer_distance(points.detach().unsqueeze(0).to(torch.float32).to('cpu'),
                                       sfm_xyz.detach().unsqueeze(0).to(torch.float32).to('cpu'), single_directional=True)
        # distance between each point in pcd initialisation and its nearest neighbor in sfm
        sfm_precision = chamfer_distance(sfm_xyz.detach().unsqueeze(0).to(torch.float32).to('cpu'),
                                        points.detach().unsqueeze(0).to(torch.float32).to('cpu'), single_directional=True)
        # distance between each point in sfm and its nearest neighbor in pcd initialisation
        sfm_loss = chamfer_distance(points.detach().unsqueeze(0).to(torch.float32).to('cpu'),
                                   sfm_xyz.detach().unsqueeze(0).to(torch.float32).to('cpu'))
        # record losses
        self.sfm_accuracy.append(sfm_accuracy[0].item())
        self.sfm_precision.append(sfm_precision[0].item())
        self.sfm_loss.append(sfm_loss[0].item())
        #self.gt_accuracy.append(gt_accuracy[0].item())
        #self.gt_precision.append(gt_precision[0].item())
        #self.gt_loss.append(gt_loss[0].item())
        #self.scale_means.append(self.depth_scales.mean())

    def plot_losses(self):
        num = 'test2'
        # sfm losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.sfm_loss, marker='o', linestyle='-', color='blue', label='Chamfer Loss')
        plt.plot(self.sfm_accuracy, marker='o', linestyle='-', color='green', label='Accuracy Loss')
        plt.plot(self.sfm_precision, marker='o', linestyle='-', color='red', label='Precision Loss')
        for i in range(len(self.sfm_loss)):
            plt.text(i, self.sfm_loss[i], f'{self.scale_means[i]}', fontsize=9, ha='right', va='bottom')
        plt.title(f'SfM Chamfer distance')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'eval_graphs/sfm_chamfer_loss_plot{num}.png')
        '''
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, marker='o', linestyle='-', color='blue', label='Chamfer Loss')
        for i in range(len(self.train_losses)):
            plt.text(i, self.train_losses[i], f'{self.scale_means[i]}', fontsize=9, ha='right', va='bottom')
        plt.title(f'Image warp losses')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'eval_graphs/train_loss_plot{num}.png')
        '''
    def render_point_cloud(self, depth_scales):
        #pcd =
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.003,
            points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )