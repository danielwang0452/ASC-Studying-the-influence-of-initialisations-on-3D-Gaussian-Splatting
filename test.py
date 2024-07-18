#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from random import randint
#from utils.loss_utils import l1_loss, ssim
from sh_utils_copy import eval_sh, SH2RGB, RGB2SH
from gaussian_renderer import render, network_gui
from typing import NamedTuple
#import sys
#from scene import Scene, GaussianModel
#from utils.general_utils import safe_state
#import uuid
#from tqdm import tqdm
#from utils.image_utils import psnr
#from argparse import ArgumentParser, Namespace
#from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
from test_camera import MiniCam, orbit_camera
from test_SimpleGaussianModel import SimpleGaussianModel
class Renderer:

    def __init__(self, sh_degree=3, white_background=True, radius=1):
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.gaussians = SimpleGaussianModel(sh_degree)
        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # init from random point cloud
        phis = np.random.random((num_pts,)) * 2 * np.pi
        costheta = np.random.random((num_pts,)) * 2 - 1
        thetas = np.arccos(costheta)
        mu = np.random.random((num_pts,))
        radius = radius * np.cbrt(mu)
        x = radius * np.sin(thetas) * np.cos(phis)
        y = radius * np.sin(thetas) * np.sin(phis)
        z = radius * np.cos(thetas)
        xyz = np.stack((x, y, z), axis=1)
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )
        self.gaussians.create_from_pcd(pcd, 10)

    def render(
            self,
            viewpoint_camera,
            scaling_modifier=1.0,
            bg_color=None,
            override_color=None,
            compute_cov3D_python=False,
            convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
                torch.zeros_like(
                    self.gaussians.get_xyz,
                    dtype=self.gaussians.get_xyz.dtype,
                    requires_grad=True,
                    device="cuda",
                )
                + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp, # None
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp, # None
        )
        # rendered_image = rendered_image.cpu().detach().clamp(0, 1).numpy()
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return rendered_image

# randomly initialise gaussians and render an image
''' Opt properties:
# training camera radius
radius: 2
# training camera fovy
fovy: 49.1 # align with zero123 rendering setting (ref: https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py#L61
# training camera min elevation
min_ver: -30
# training camera max elevation
max_ver: 30
H: 800
W: 800
sh_degree: 0
'''
def test():
    params = {
        # defined in image.yaml
        'radius': 2,
        'fovy': 49.1,
        'min_ver': -30,
        'max_ver': 30,
        'H': 800,
        'W': 800,
        'sh_degree': 0,
        # defined in OrbitCamera Class
        'near' : 0.01,
        'far' : 100,
        # extra self defined params
        'elevation' : 0, # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
        'azimuth' : 0,   # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
        'render_resolution' : 128
    }
    params['fovx'] = 2 * np.arctan(np.tan(params['fovy'] / 2) * params['W'] / params['H'])
    # initialise renderer
    renderer = Renderer(sh_degree=3, white_background=True, radius=0.5) # radius containing gaussians
    # initialise gaussians, stored in renderer
    renderer.initialize(num_pts=1, radius=0.5)
    # initialise camera
    pose = orbit_camera(elevation=params['elevation'],
                        azimuth=params['azimuth'],
                        radius=5
                       )  # return: [4, 4], camera pose matrix
    viewpoint_camera = MiniCam(pose, params['render_resolution'], params['render_resolution'],
                               params['fovy'], params['fovx'], params['near'], params['far'])
    # render
    rendered_image = renderer.render(
            viewpoint_camera,
            scaling_modifier=1.0,
            bg_color=None,
            override_color=None,
            compute_cov3D_python=False,
            convert_SHs_python=False,
    )
    # display image - shape (C, H, W)
    image = rendered_image.cpu().detach().clamp(0, 1).permute((1, 2, 0)).numpy()
    print(image)
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()

    image_pil = Image.fromarray((image * 255).astype('uint8'))
    image_pil.save('rendered_image.png')
    image_pil.show()

    return

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

if __name__ == "__main__":
    test()
