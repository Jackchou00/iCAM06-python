import numpy as np


def XYZ_to_sRGB(XYZ):
    XYZ = np.clip(XYZ / 100, 0, 1)
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    RGB = np.dot(XYZ, M.T)
    RGB = np.clip(RGB, 0, 1)
    RGB = np.where(RGB <= 0.0031308, 12.92 * RGB, 1.055 * RGB ** (1 / 2.4) - 0.055)
    return RGB