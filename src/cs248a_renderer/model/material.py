from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple
from pathlib import Path
import slangpy as spy
from pyglm import glm
import numpy as np
from PIL import Image


@dataclass
class PhysicsBasedMaterialTextureBuf:
    albedo: spy.NDBuffer


class FilteringMethod(Enum):
    NEAREST = 0
    BILINEAR = 1
    BILINEAR_DISCRETIZED_LEVEL = 2
    TRILINEAR = 3


@dataclass
class MaterialField[T]:
    uniform_value: T | None = None
    use_texture: bool = False
    filtering_method: FilteringMethod = FilteringMethod.NEAREST
    textures: List[np.ndarray] = field(default_factory=list)

    def __init__(self, uniform_value: T | None = None, use_texture: bool = False, filtering_method: FilteringMethod = FilteringMethod.NEAREST, texture_map_path: str = None):
        self.uniform_value = uniform_value
        self.use_texture = use_texture
        self.filtering_method = filtering_method
        self.MAX_MIP_LEVELS = 8
        self.textures = []
        if texture_map_path is None:
            if self.use_texture:
                raise ValueError("Texture map path is required for ALBEDO_TEXTURE material")
        else:
            self.load_texture_from_image(texture_map_path)

    def generate_mipmaps(self, base_texture: np.ndarray) -> None:
        """Generate mipmaps from a base texture image.

        :param base_texture: The base texture as numpy array (H, W, C)
        """
        # This is level 0 mipmap. It is the original texture. Populate this list
        #  with other mipmap levels, in increasing order of levels.
        textures = [base_texture]
        # TODO: Student implementation starts here.
        

        cur_mipmap = base_texture
        for i in range(1, self.MAX_MIP_LEVELS):
            h, w, c = cur_mipmap.shape
            cur_mipmap = cur_mipmap[:(h // 2) * 2, :(w // 2) * 2]
            reshaped_mipmap = cur_mipmap.reshape(h//2, 2, w//2, 2, c)
            cur_mipmap = reshaped_mipmap.mean(axis=(1, 3))
            textures.append(cur_mipmap)
        # TODO: Student implementation ends here.

        self.textures = textures

    def load_texture_from_image(self, image_path: str | Path) -> None:
        """Load a texture from an image file and generate mipmaps.

        :param image_path: Path to the image file (PNG, JPG, etc.)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        pil_img = Image.open(image_path)
        # Only keep the rgb channels
        pil_img = pil_img.convert("RGB")
        # Convert to numpy array and normalize to [0, 1]
        texture_array = np.array(pil_img).astype(np.float32) / 255.0

        # Generate mipmaps
        self.generate_mipmaps(texture_array)

    def get_this(self, offset: int) -> Tuple[Dict, int]:
        mipmaps = []
        for i in range(self.MAX_MIP_LEVELS):
            if i >= len(self.textures):
                mipmaps.append(
                    {
                        "size": [0, 0],
                        "offset": offset,
                    }
                )
            else:
                texture = self.textures[i]
                mipmaps.append(
                    {
                        "size": [texture.shape[1], texture.shape[0]],
                        "offset": offset,
                    }
                )
                offset += texture.shape[0] * texture.shape[1]
        return {
            "uniformData": self.uniform_value,
            "useTexture": self.use_texture,
            "filteringMethod": self.filtering_method.value,
            "mipmap": mipmaps,
            "totalLevels": len(self.textures),
        }, offset


@dataclass
class PhysicsBasedMaterial:
    albedo: MaterialField[glm.vec3] = field(
        default_factory=lambda: MaterialField(glm.vec3(1.0, 0.0, 1.0))
    )

    def get_this(self, offset: int) -> Tuple[Dict, int]:
        albedo_data, offset = self.albedo.get_this(offset)
        return {
            "albedo": albedo_data,
        }, offset


def create_material_buf(
    module: spy.Module, materials: List[PhysicsBasedMaterial]
) -> Tuple[spy.NDBuffer, PhysicsBasedMaterialTextureBuf]:
    device = module.device
    material_buf = spy.NDBuffer(
        device=device,
        dtype=module.PhysicsBasedMaterial.as_struct(),
        shape=(max(len(materials), 1),),
    )
    albedo_textures = []
    cursor = material_buf.cursor()
    offset = 0
    for idx, material in enumerate(materials):
        material_data, offset = material.get_this(offset)
        for level_texture in material.albedo.textures:
            albedo_textures.append(level_texture.astype(np.float32).reshape(-1, 3))
        cursor[idx].write(material_data)
    cursor.apply()
    # Concatenate all albedo textures into a single buffer.
    np_alebdo_texture = (
        np.concatenate(albedo_textures, axis=0)
        if albedo_textures
        else np.zeros((1, 3), dtype=np.float32)
    )
    albedo_tex_buf = spy.NDBuffer(
        device=device,
        dtype=module.float3,
        shape=(np_alebdo_texture.shape[0],),
    )
    albedo_tex_buf.copy_from_numpy(np_alebdo_texture)
    physics_based_material_texture_buf = PhysicsBasedMaterialTextureBuf(
        albedo=albedo_tex_buf
    )
    return material_buf, physics_based_material_texture_buf
