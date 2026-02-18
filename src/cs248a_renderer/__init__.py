from typing import List
from pathlib import Path
import platform

import slangpy as spy
import slangpy_nn as nn

SHADER_PATH = Path(__file__).parent / "slang_shaders"

_system = platform.system()
if _system == "Darwin":
    DEVICE_TYPE = spy.DeviceType.metal
elif _system in ("Windows", "Linux"):
    DEVICE_TYPE = spy.DeviceType.vulkan
else:
    # Default to vulkan for unknown/other platforms
    DEVICE_TYPE = spy.DeviceType.vulkan


def setup_device(notebook_shader_paths: List[Path]) -> spy.Device:
    # CS 248A Shaders
    shader_paths = [
        SHADER_PATH.absolute(),
    ]
    # SlangNN Shaders
    shader_paths.extend(nn.slang_include_paths())
    # Extra Notebook Shaders
    shader_paths.extend([p.absolute() for p in notebook_shader_paths])
    device = spy.create_device(include_paths=shader_paths, type=DEVICE_TYPE)
    return device


class RendererModules:
    math_module: spy.Module
    texture_module: spy.Module
    material_module: spy.Module
    primitive_module: spy.Module
    model_module: spy.Module
    renderer_module: spy.Module

    def __init__(self, device: spy.Device):
        self.math_module = spy.Module.load_from_file(
            device=device,
            path="math.slang",
        )
        self.texture_module = spy.Module.load_from_file(
            device=device,
            path="texture.slang",
            link=[self.math_module],
        )
        self.material_module = spy.Module.load_from_file(
            device=device,
            path="material.slang",
            link=[self.math_module],
        )
        self.primitive_module = spy.Module.load_from_file(
            device=device,
            path="primitive.slang",
            link=[self.math_module, self.texture_module],
        )
        self.model_module = spy.Module.load_from_file(
            device=device,
            path="model.slang",
            link=[self.math_module],
        )
        self.renderer_module = spy.Module.load_from_file(
            device=device,
            path="renderer.slang",
            link=[
                self.math_module,
                self.texture_module,
                self.material_module,
                self.primitive_module,
                self.model_module,
            ],
        )
