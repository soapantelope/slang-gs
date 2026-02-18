"""Dataset generation script for volumetric renderer outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyglm import glm
import slangpy as spy
from tqdm import tqdm
from PIL import Image

from cs248a_renderer.model.volumes import VolumeProperties
from cs248a_renderer.renderer.volume_renderer import VolumeRenderer
from cs248a_renderer.view_model.scene_manager import SceneManager


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _create_device(include_shaders: Optional[List[Path]] = None) -> spy.Device:
    repo_root = _resolve_repo_root()
    if include_shaders is None:
        include_shaders = []

    default_paths = [
        repo_root / "src" / "slang_volumetric_renderer" / "slang_shaders",
        repo_root / "notebooks" / "volume-renderer" / "shaders",
    ]

    include_paths = []
    for path in [*(include_shaders or []), *default_paths]:
        if path.exists():
            include_paths.append(path.resolve())

    if not include_paths:
        raise FileNotFoundError(
            "No shader include paths found. Please check repository layout."
        )

    return spy.create_device(include_paths=include_paths)


def _setup_renderer(
    volume_path: Path, width: int, height: int
) -> Tuple[VolumeRenderer, SceneManager, spy.Texture]:
    device = _create_device()
    output_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
    )

    renderer = VolumeRenderer(device=device, render_texture=output_texture)
    scene_manager = SceneManager()
    scene_manager.create_volume_from_numpy(
        volume_path=volume_path,
        properties=VolumeProperties(
            voxel_size=0.01,
            pivot=(0.5, 0.5, 0.5),
            albedo=(1.0, 1.0, 1.0),
        ),
    )

    return renderer, scene_manager, output_texture


def _render_sample(
    renderer: VolumeRenderer,
    scene_manager: SceneManager,
    render_target: spy.Texture,
    camera_distance: float,
):
    scene = scene_manager.volume_scene
    assert scene is not None

    pos_arr = np.random.normal(size=(3,))
    pos_arr = pos_arr / np.linalg.norm(pos_arr) * camera_distance
    scene.camera.transform.position = glm.vec3(*pos_arr)
    scene.camera.transform.rotation = glm.quatLookAt(
        glm.normalize(scene.camera.transform.position * -1.0),
        glm.vec3(0.0, 1.0, 0.0),
    )

    renderer.render_with_cache(
        scene=scene, view_mat=scene.camera.view_matrix(), fov=scene.camera.fov
    )
    img_arr = render_target.to_numpy()

    return {
        "img_arr": img_arr,
        "position": np.array(scene.camera.transform.position),
        "rotation": np.array(scene.camera.transform.rotation),
        "fov": scene.camera.fov,
    }


def _save_image(image: np.ndarray, path: Path) -> None:
    clipped = np.clip(image, 0.0, 1.0)
    png_data = (clipped * 255.0).round().astype(np.uint8)
    png_data = np.flipud(png_data)
    Image.fromarray(png_data, mode="RGBA").save(path)


def generate_dataset(
    output_dir: Path,
    num_images: int,
    volume_path: Path,
    width: int = 512,
    height: int = 512,
    camera_distance: float = 4.0,
    seed: Optional[int] = None,
    metadata_filename: str = "metadata.json",
):
    """Render a dataset to PNG images with camera metadata."""

    output_dir = Path(output_dir)
    volume_path = Path(volume_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    renderer, scene_manager, render_target = _setup_renderer(
        volume_path=volume_path, width=width, height=height
    )
    renderer.load_volume(scene_manager.volume_scene)

    metadata: List[Dict[str, object]] = []

    for index in tqdm(range(num_images), desc="Rendering dataset"):
        sample = _render_sample(
            renderer=renderer,
            scene_manager=scene_manager,
            render_target=render_target,
            camera_distance=camera_distance,
        )
        file_name = f"image_{index:04d}.png"
        _save_image(sample["img_arr"], output_dir / file_name)
        metadata.append(
            {
                "file_name": file_name,
                "position": sample["position"].astype(float).tolist(),
                "rotation": sample["rotation"].astype(float).tolist(),
                "fov": float(sample["fov"]),
            }
        )

    metadata_path = output_dir / metadata_filename
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump({"images": metadata}, handle, indent=2)

    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate volumetric rendering dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for rendered PNGs",
    )
    parser.add_argument(
        "--volume-path",
        type=Path,
        default=_resolve_repo_root() / "resources" / "bunny_cloud.npy",
        help="Path to volume numpy file",
    )
    parser.add_argument(
        "--num-images", type=int, default=256, help="Number of images to render"
    )
    parser.add_argument("--width", type=int, default=512, help="Render target width")
    parser.add_argument("--height", type=int, default=512, help="Render target height")
    parser.add_argument(
        "--camera-distance", type=float, default=4.0, help="Camera distance from origin"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--metadata-name", type=str, default="metadata.json", help="Metadata file name"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        volume_path=args.volume_path,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        seed=args.seed,
        metadata_filename=args.metadata_name,
    )


if __name__ == "__main__":
    main()
