"""
Volume Wizard Menu Item and Popup.
"""

import asyncio
from enum import Enum
import logging
from pathlib import Path
from typing import Tuple, Unpack
from imgui_bundle import imgui, portable_file_dialogs as pfd
from reactivex.subject import Subject
from slangpy_imgui_bundle.render_targets.render_target import RenderTarget, RenderArgs
from slangpy_imgui_bundle.utils.file_dialog import async_file_dialog
import slangpy as spy

from cs248a_renderer.model.volumes import VolumeProperties
from cs248a_renderer.model.nerf import NeRFProperties
from cs248a_renderer.view_model.scene_manager import SceneManager


logger = logging.getLogger(__name__)


class VolumeWizardArgs(RenderArgs):
    open: Subject[None]
    scene_manager: SceneManager


class CreateType(Enum):
    VOLUME_FROM_FILE = "Volume From Numpy File"
    NERF_FROM_FILE = "NeRF From Numpy File"


class SceneWizard(RenderTarget):
    _open: bool = False

    def __init__(self, **kwargs: Unpack[VolumeWizardArgs]) -> None:
        super().__init__(**kwargs)
        self._scene_manager = kwargs["scene_manager"]

        def on_open(_: None) -> None:
            self._open = True

        kwargs["open"].subscribe(on_open)

        self._reset()

    def render(self, time: float, delta_time: float) -> None:
        if self._open:
            imgui.open_popup("Scene Wizard")
            self._open = False
            self._reset()
        if imgui.begin_popup_modal(
            "Scene Wizard", flags=imgui.WindowFlags_.always_auto_resize.value
        )[0]:
            # Create type selection.
            _, create_type_idx = imgui.combo(
                "Create Type",
                list(CreateType).index(self.create_type),
                [ct.value for ct in CreateType],
            )
            self.create_type = list(CreateType)[create_type_idx]

            imgui.separator()

            if self.create_type == CreateType.VOLUME_FROM_FILE:
                # Create from file parameters.
                _, _ = imgui.input_text(
                    "Volume Path",
                    str(self.volume_path),
                    flags=imgui.InputTextFlags_.read_only.value,
                )
                if imgui.button("Choose File"):
                    asyncio.create_task(self._choose_file())

                # Voxel size.
                _, self.voxel_size = imgui.input_float("Voxel Size", self.voxel_size)
                # Ensure minimum voxel size.
                self.voxel_size = max(0.001, self.voxel_size)

                _, new_pivot = imgui.drag_float3(
                    "Pivot", list(self.pivot), 0.01, 0.0, 1.0
                )
                self.pivot = (new_pivot[0], new_pivot[1], new_pivot[2])

            elif self.create_type == CreateType.NERF_FROM_FILE:
                # Create from file parameters.
                _, _ = imgui.input_text(
                    "NeRF Path",
                    str(self.volume_path),
                    flags=imgui.InputTextFlags_.read_only.value,
                )
                if imgui.button("Choose File"):
                    asyncio.create_task(self._choose_file())

                # Bounding box size
                _, new_bbox = imgui.drag_float3(
                    "Bounding Box Size", list(self.bounding_box_size), 0.1, 0.0, 100.0
                )
                self.bounding_box_size = (new_bbox[0], new_bbox[1], new_bbox[2])

                _, new_pivot = imgui.drag_float3(
                    "Pivot", list(self.pivot), 0.01, 0.0, 1.0
                )
                self.pivot = (new_pivot[0], new_pivot[1], new_pivot[2])

            if imgui.button("Create"):
                # Placeholder for volume creation logic.
                self._create_scene()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Close"):
                imgui.close_current_popup()
            imgui.end_popup()

    def _reset(self) -> None:
        self.create_type = CreateType.VOLUME_FROM_FILE
        # Empty volume parameters.
        self.volume_size = (32, 32, 32, 1)
        # Create from file parameters.
        self.volume_path = Path.home()

        self.voxel_size = 0.01
        self.bounding_box_size = (2.0, 2.0, 2.0)
        self.pivot = (0.5, 0.5, 0.5)

    async def _choose_file(self) -> None:
        files = await async_file_dialog(
            title="Open File",
            default_path=str(Path.cwd()),
            filters=["NumPy Files", "*.npy *.npz"],
            options=pfd.opt.none,
        )
        if files:
            self.volume_path = Path(files[0])
            logger.info("Selected volume file: %s", self.volume_path)

    def _create_scene(self) -> None:
        volume_properties = VolumeProperties(
            voxel_size=self.voxel_size, pivot=self.pivot, albedo=(1.0, 1.0, 1.0)
        )
        if self.create_type == CreateType.VOLUME_FROM_FILE:
            try:
                self._scene_manager.create_volume_from_numpy(
                    volume_path=self.volume_path,
                    properties=volume_properties,
                )
                logger.info("Created volume scene from file %s", self.volume_path)
            except Exception as e:
                logger.error("Failed to create volume from file: %s", e)
        elif self.create_type == CreateType.NERF_FROM_FILE:
            nerf_properties = NeRFProperties(
                bounding_box_size=self.bounding_box_size,
                pivot=self.pivot,
            )
            try:
                assert self._device is not None
                device_module = self._device.load_module("model.slang")
                module = spy.Module(device_module)
                self._scene_manager.create_nerf_from_numpy(
                    module=module,
                    nerf_path=self.volume_path,
                    properties=nerf_properties,
                )
                logger.info("Created NeRF scene from file %s", self.volume_path)
            except Exception as e:
                logger.error("Failed to create NeRF from file: %s", e)
