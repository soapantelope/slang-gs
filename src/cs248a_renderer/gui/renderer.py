from typing import Tuple, Unpack
from imgui_bundle import ImVec2, imgui, imgui_ctx, imgui_tex_inspect
from reactivex import Observable
from reactivex.subject import Subject
import slangpy as spy
from slangpy_imgui_bundle.render_targets.window import Window, WindowArgs


class RendererWindowArgs(WindowArgs):
    render_texture: Observable[Tuple[spy.Texture, int]]
    render_request: Subject[None]


class RendererWindow(Window):
    _render_texture: spy.Texture
    _render_texture_id: int

    _render_request: Subject[None]
    _real_time_rendering: bool = False

    def __init__(self, **kwargs: Unpack[RendererWindowArgs]) -> None:
        super().__init__(**kwargs)

        def update_texture(texture: Tuple[spy.Texture, int]):
            self._render_texture, self._render_texture_id = texture

        kwargs["render_texture"].subscribe(update_texture)

        self._render_request = kwargs["render_request"]

    def render_window(self, time: float, delta_time: float, open: bool | None) -> bool:
        window_flags = imgui.WindowFlags_.menu_bar.value
        with imgui_ctx.begin("Renderer", p_open=open, flags=window_flags) as window:
            # Menu bar.
            if imgui.begin_menu_bar():
                if imgui.menu_item_simple("Render"):
                    self._render_request.on_next(None)
                imgui.end_menu_bar()

            # Get the available content space.
            content_region_avail = imgui.get_content_region_avail()
            imgui_tex_inspect.begin_inspector_panel(
                "Render Result Inspector",
                self._render_texture_id,
                ImVec2(self._render_texture.width, self._render_texture.height),
                flags=imgui_tex_inspect.InspectorFlags_.flip_y.value,
                size=imgui_tex_inspect.SizeIncludingBorder(content_region_avail),
            )
            imgui_tex_inspect.end_inspector_panel()

            return window.opened
