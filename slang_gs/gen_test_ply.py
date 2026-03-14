import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
from pathlib import Path

def logit(x):
    return np.log(x / (1.0 - x))

points = pd.DataFrame({
    'x': [-0.2, 0.3, 0.0],
    'y': [0.0, 0.0, 2.0],
    'z': [-0.1, -1, 0.0],
    'rot_0': [1.0, 1.0, 1.0],
    'rot_1': [0.3, 0.2, 0.2],
    'rot_2': [0.9, 0.0, 0.4],
    'rot_3': [0.0, 0.2, 0.9],
    'scale_0': np.log([0.9, 0.6, 0.3]),
    'scale_1': np.log([1.0, 0.6, 0.6]),
    'scale_2': np.log([0.3, 0.2, 0.1]),
    'f_dc_0': [1.0, 0.0, 0.0],
    'f_dc_1': [0.0, 1.0, 0.0],
    'f_dc_2': [0.0, 0.0, 1.0],
    'opacity': logit(np.array([0.99, 0.99, 0.99])),
})

out = Path(__file__).parent.parent / "resources" / "test.ply"
out.parent.mkdir(parents=True, exist_ok=True)
PyntCloud(points).to_file(str(out))
print(f"Wrote {out}")
