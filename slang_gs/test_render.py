import argparse
import torch
import math
from pathlib import Path
from matplotlib import pyplot as plt

from gaussian_model import GaussianModel
from viewpoint_camera import ViewpointCamera
from init_3dgs import render

parser = argparse.ArgumentParser()
parser.add_argument("ply", type=Path, help="path to .ply file")
args = parser.parse_args()

pc = GaussianModel(str(args.ply))
print(f"Loaded {pc.get_xyz.shape[0]} gaussians from {args.ply.name}")

cam = ViewpointCamera(
    position=[0.0, 0.0, 5.0],
    target=[0.0, 0.0, 0.0],
    up=[0.0, 1.0, 0.0],
    fov_y=math.radians(60),
    image_width=512,
    image_height=512,
)

bg_color = torch.zeros(3, device="cuda")

out = render(cam, pc, pipe=None, bg_color=bg_color)

img = out["render"].detach().cpu().numpy()
print(f"rendered image shape: {img.shape}")

plt.imshow(img[:, :, :3].clip(0, 1))
plt.title("test render")
plt.axis("off")
plt.show()
