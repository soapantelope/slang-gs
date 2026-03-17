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
parser.add_argument("--frames", type=int, default=None, help="number of frames (overridden by --duration if set)")
parser.add_argument("--duration", type=float, default=None, help="video duration in seconds (sets frames = duration * fps, slower spiral)")
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--radius", type=float, default=5.0, help="orbit radius (distance from centroid)")
parser.add_argument("--loops", type=float, default=2.0, help="number of full circles while spiraling top to bottom")
parser.add_argument("--elevation-top", type=float, default=80.0, help="elevation at start in degrees (90=top, 0=horizon)")
parser.add_argument("--elevation-bottom", type=float, default=-80.0, help="elevation at end in degrees (-90=bottom)")
parser.add_argument("--fov", type=float, default=60.0, help="vertical FOV in degrees")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("-o", "--output", type=Path, default=None, help="output mp4 path (default: <ply_stem>_spiral.mp4)")
args = parser.parse_args()

if args.duration is not None:
    args.frames = int(args.duration * args.fps)
elif args.frames is None:
    args.frames = 120

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

# Spherical spiral: small circles at top/bottom, large circle at sides (like standard splat previews)
elev_top_rad = math.radians(args.elevation_top)
elev_bottom_rad = math.radians(args.elevation_bottom)
for i in range(args.frames):
    t = i / (args.frames - 1) if args.frames > 1 else 1.0
    # Elevation: start near top, end near bottom
    elev = elev_top_rad + t * (elev_bottom_rad - elev_top_rad)
    # Azimuth: full circles as we move top -> bottom
    theta = 2.0 * math.pi * args.loops * t
    # Sphere: x = r*cos(elev)*cos(theta), y = r*sin(elev), z = r*cos(elev)*sin(theta)
    r_cos = args.radius * math.cos(elev)
    r_sin = args.radius * math.sin(elev)
    pos = [
        centroid[0] + r_cos * math.cos(theta),
        centroid[1] + r_sin,
        centroid[2] + r_cos * math.sin(theta),
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
