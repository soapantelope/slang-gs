from __future__ import annotations

import torch
import math
import glm

from renderer import Renderer
from pathlib import Path
from gaussian_model import GaussianModel
import slangtorch

SHADER_PATH = Path(__file__).parent / "slang_shaders"

_cuda_module = None

def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        _cuda_module = slangtorch.loadModule(str(SHADER_PATH / "renderer.slang"))
    return _cuda_module

# to insert with 3DGS
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, separate_sh=False, override_color=None, use_trained_exp=False):
    renderer = Renderer()
    renderer_cuda_module = _get_cuda_module()

    positions =  pc.get_xyz
    rotations = pc.get_rotation
    scales = pc.get_scaling
    colors = pc.get_features_dc.squeeze(1)  # raw f_dc_0, 1, and 2 for now
    opacities = pc.get_opacity
    num_gaussians = positions.shape[0]

    focal_x = viewpoint_camera.image_width / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    focal_y = viewpoint_camera.image_height / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))

    rendered_image, viewspace_points, radii = renderer.apply(
        renderer_cuda_module,
        positions,
        rotations,
        scales,
        colors,
        opacities,
        num_gaussians,
        viewpoint_camera.world_view_transform,
        viewpoint_camera.projection_matrix,
        focal_x,
        focal_y,
        int(viewpoint_camera.image_height),
        int(viewpoint_camera.image_width),
        glm.ivec2(64, 64)
    )

    out = {
        "render": rendered_image,
        "viewspace_points": viewspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
    }
    
    return out
