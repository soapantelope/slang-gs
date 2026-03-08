import slangpy as spy
from pyglm import glm
from matplotlib import pyplot as plt
import numpy as np

from cs248a_renderer import setup_device, RendererModules
from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.gaussian_splat import GaussianSplat
from cs248a_renderer.renderer.core_renderer import Renderer
from pathlib import Path

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
renderer._ambientColor = glm.vec4(0.0, 0.0, 0.0, 0.0)

cam = PerspectiveCamera()
cam_pos = glm.vec3(0, 3, -2)
cam.transform.position = cam_pos
cam.transform.rotation = glm.quatLookAt(glm.normalize(-cam_pos), glm.vec3(0.0, 1.0, 0.0))

gaussian = GaussianSplat(device=device, path=Path("../resources/bonsai.ply"))
renderer.load_gaussian(gaussian=gaussian)

renderer.sqrt_spp = 1
renderer.render_gaussians(
    cam.view_matrix(),
    cam.projection_matrix(OUTPUT_IMG_SIZE[0], OUTPUT_IMG_SIZE[1]),
    glm.ivec2(32, 32),
    cam.fov
)
img_data = np.flipud(output_image.to_numpy())
plt.imshow(img_data)
plt.imsave('output.png', img_data)

plt.axis('off')
plt.show()