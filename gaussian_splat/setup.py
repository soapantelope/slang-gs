import slangpy as spy
from pyglm import glm
from matplotlib import pyplot as plt
import numpy as np

from cs248a_renderer import setup_device, RendererModules
from cs248a_renderer.model.scene import Scene
from cs248a_renderer.renderer.core_renderer import Renderer
from cs248a_renderer.model.volumes import DenseVolume
from cs248a_renderer.model.transforms import Transform3D

device = setup_device([])
renderer_modules = RendererModules(device)

OUTPUT_IMG_SIZE = (512, 512)
output_image = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    usage=spy.TextureUsage.unordered_access,
    width=OUTPUT_IMG_SIZE[0],
    height=OUTPUT_IMG_SIZE[1],
)

renderer = Renderer(
    device=device,
    render_texture=output_image,
    render_modules=renderer_modules
)
renderer.sqrt_spp = 1

scene = Scene()
cam = scene.camera
cam_pos = glm.vec3(1, 1, 1) * 3
cam.transform.position = cam_pos
cam.transform.rotation = glm.quatLookAt(glm.normalize(-cam_pos), glm.vec3(0.0, 1.0, 0.0))

lego_volume = np.load("../resources/lego_volume.npy")
volume = DenseVolume(
    name="volume",
    transform=Transform3D(
        position=glm.vec3(0.0, 0.0, 0.0),
        rotation=glm.angleAxis(glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0)),
        scale=glm.vec3(1.0, 1.0, 1.0),
    ),
    data=lego_volume.astype(np.float32),
    properties={
        "pivot": (0.5, 0.5, 0.5),
        "voxel_size": 0.02,
    }
)
renderer.load_volume(volume=volume)
renderer.sqrt_spp = 1
renderer.render(
    scene.camera.view_matrix(),
    scene.camera.fov
)
plt.imshow(np.flipud(output_image.to_numpy()))
plt.axis('off')
plt.show()