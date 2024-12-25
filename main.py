import numpy as np
from matplotlib import pyplot as plt
from fastbiliateral_blur import fast_bilateral_filter, blur
from iCAM06 import iCAM06_CAT, iCAM06_TC, iCAM06_IPT
import cv2


def XYZ_to_RGB(XYZ):
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


def LocalContrast(detail, base):
    La = 0.2 * base[:, :, 1]
    k = 1.0 / (5 * La + 1)
    FL = 0.2 * k ** 4 * (5 * La) + 0.1 * (1 - k ** 4) ** 2 * (5 * La) ** (1 / 3)
    FL_rep = np.stack([FL, FL, FL], axis=2)
    detail_a = np.power(detail, np.power((FL_rep + 0.8), 0.25))
    return detail_a


def main():
    # read mat
    # im_array, _, _ = read_mat()
    im_array = np.load('xyz.npy')
    
    # Image decomposition
    base_layer = np.zeros_like(im_array)
    detail_layer = np.zeros_like(im_array)
    for i in range(3):
        base_layer[:, :, i], detail_layer[:, :, i] = fast_bilateral_filter(im_array[:, :, i])

    # Chromatic adaptation
    white = blur(im_array, 2)
    XYZ_adapt = iCAM06_CAT(base_layer, white)
    
    # Tone compression
    white = blur(im_array, 3)
    XYZ_tc = iCAM06_TC(XYZ_adapt, white, 0.7)
    
    # Image attribute adjustments
    XYZ_d = XYZ_tc * LocalContrast(detail_layer, base_layer)
    XYZ_p = iCAM06_IPT(XYZ_d, base_layer, 1.0)
    
    # Convert XYZ to RGB
    RGB_p = XYZ_to_RGB(XYZ_p)
    
    plt.imshow(RGB_p)
    plt.show()
    
    # Convert RGB from RGB to BGR for OpenCV
    RGB_p_bgr = cv2.cvtColor((RGB_p * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # Save the image
    cv2.imwrite('output.jpg', RGB_p_bgr)
    
    
if __name__ == "__main__":
    main()
