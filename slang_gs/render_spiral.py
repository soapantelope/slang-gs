# i asked chatgpt to generate a script to render a spiral video around a gaussian PLY using my renderer!

import argparse
import math
import subprocess
import torch
import numpy as np
from pathlib import Path

from gaussian_model import GaussianModel
from viewpoint_camera import ViewpointCamera
from init_3dgs import render

parser = argparse.ArgumentParser(description="Render a spiral video around a gaussian PLY")
parser.add_argument("ply", type=Path, help="path to .ply file")
parser.add_argument("--frames", type=int, default=120, help="number of frames")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--radius", type=float, default=5.0, help="orbit radius")
parser.add_argument("--height-amplitude", type=float, default=1.5, help="vertical oscillation amplitude")
parser.add_argument("--loops", type=int, default=2, help="number of vertical oscillations per revolution")
parser.add_argument("--fov", type=float, default=60.0, help="vertical FOV in degrees")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("-o", "--output", type=Path, default=None, help="output mp4 path (default: <ply_stem>_spiral.mp4)")
args = parser.parse_args()

output_path = args.output or args.ply.with_name(args.ply.stem + "_spiral.mp4")

pc = GaussianModel(str(args.ply))
print(f"Loaded {pc.get_xyz.shape[0]} gaussians from {args.ply.name}")

centroid = pc.get_xyz.mean(dim=0).cpu().numpy()
target = centroid.tolist()

# Pipe raw RGB frames to ffmpeg — no per-frame disk I/O
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{args.width}x{args.height}",
    "-r", str(args.fps),
    "-i", "pipe:0",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    str(output_path),
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

frame_bytes = args.width * args.height * 3
for i in range(args.frames):
    t = i / args.frames
    theta = 2.0 * math.pi * t
    y_offset = args.height_amplitude * math.sin(2.0 * math.pi * args.loops * t)

    pos = [
        centroid[0] + args.radius * math.cos(theta),
        centroid[1] + y_offset,
        centroid[2] + args.radius * math.sin(theta),
    ]

    cam = ViewpointCamera(
        position=pos,
        target=target,
        up=[0.0, 1.0, 0.0],
        fov_y=math.radians(args.fov),
        image_width=args.width,
        image_height=args.height,
    )

    bg_color = torch.zeros(3, device="cuda")
    out = render(cam, pc, pipe=None, bg_color=bg_color)
    img = out["render"].detach().cpu().numpy()[:, :, :3].clip(0, 1)
    rgb = (img * 255.0).astype(np.uint8)
    proc.stdin.write(rgb.tobytes())
    print(f"\r  frame {i + 1}/{args.frames}", end="", flush=True)

proc.stdin.close()
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
print(f"\nSaved {output_path}")
