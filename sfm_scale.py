import numpy as np
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import cv2
import json
from read_write_model import *
import os
import collections
import numpy as np
import struct
import argparse
import torchvision
import torch
from test_depth_anything_v2 import get_single_depth_map
from PIL import Image
try:
    from scene import scene_name
except:
    scene_name = 'garden2'

def PILtoTorch(pil_image, scale):
    resolution = (int(pil_image.size[0]/scale), int(pil_image.size[1]/scale))
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_scales(key, cameras, images, points3d_ordered):
    scale=1
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    #pts_idx = images_metas[key].point3D_ids
    pts_idx = image_meta.point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2]
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    #invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    gt_image = Image.open(f'colmaps/{scene_name}/images/{image_meta.name}')
    # resize img
    resized_gt = 255*PILtoTorch(gt_image, scale=scale).permute((1, 2, 0)).numpy()
    invmonodepthmap = get_single_depth_map(resized_gt)
    invmonodepthmap = torch.clamp_min(torch.tensor(invmonodepthmap),
                    min=0.1).numpy()
    #invmonodepthmap = 0.01*torch.ones(resized_gt.shape).numpy()

    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    #invmonodepthmap = invmonodepthmap.astype(np.float32) / (2 ** 16)
    s = invmonodepthmap.shape[0] / (cam_intrinsic.height)
    maps = (valid_xys * s).astype(np.float32)
    valid = (
            (maps[..., 0] >= 0) *
            (maps[..., 1] >= 0) *
            (maps[..., 0] < cam_intrinsic.width * s) *
            (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        print(maps.shape)
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)[..., 0]
        print(invcolmapdepth.mean())
        print(invmonodepth.mean())
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    print(scale, offset)
    metric_invdepth = scale*invmonodepthmap+offset
    #return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}
    return image_meta.name, metric_invdepth

def get_sfm_depth():
    cam_intrinsics, images_metas, points3d = read_model(path="/Users/danielwang/PycharmProjects/gaussian-splatting-test/colmaps/garden2/sparse/0/"
                                                        , ext=f".bin")
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs
    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    all_sfm_depths = []
    for key in images_metas:
        print(key)
        img_name, metric_invdepth = get_scales(key, cam_intrinsics, images_metas, points3d_ordered)
        all_sfm_depths.append((img_name, metric_invdepth))
    return all_sfm_depths

#get_sfm_depth()