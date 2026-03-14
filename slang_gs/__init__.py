import torch
import math
import glm

from cs248a_renderer import setup_device, RendererModules
from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.gaussian_splat import GaussianSplat
from cs248a_renderer.renderer.core_renderer import Renderer
from pathlib import Path

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    renderer = Renderer()

    positions =  pc.get_xyz
    rotations = pc.get_rotation
    scales = pc.get_scaling
    colors = pc.get_features_dc.squeeze(1)  # raw f_dc_0, 1, and 2 for now
    opacities = pc.get_opacity
    num_gaussians = positions.shape[0]

    renderer.load_gaussians(positions, rotations, scales, colors, opacities, num_gaussians)

    focal_x = viewpoint_camera.image_width / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    focal_y = viewpoint_camera.image_height / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))

    rendered_image, viewspace_points, radii = renderer.render_gaussians(
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        focal_length_x=focal_x,
        focal_length_y=focal_y,
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        num_tiles=glm.ivec2(32, 32)
    )

    out = {
        "render": rendered_image,
        "viewspace_points": viewspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
    }
    
    return out
