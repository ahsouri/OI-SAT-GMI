import numpy as np

grid = np.meshgrid(
    np.linspace(0, 1, 512),
    np.linspace(0, 1, 512)
)

print(np.array(1.314143141).astype('float16'))