from dataclasses import dataclass


@dataclass
class RayMarcherConfig:
    max_steps: int = 1024
    step_size: float = 0.01
    density_scale: float = 10.0
