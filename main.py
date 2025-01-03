import numpy as np
import numexpr as ne
from fastbiliateral_blur import bilateral_filter, blur
from iCAM06 import iCAM06_CAT, iCAM06_TC, iCAM06_IPT
from CAT02 import img_CAT02_to_D65
from TC import img_TC
from IPT import IPT
import cv2


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


def LocalContrast(detail, base):
    La = 0.2 * base[:, :, 1]
    k = 1.0 / (5 * La + 1)
    # FL = 0.2 * k ** 4 * (5 * La) + 0.1 * (1 - k ** 4) ** 2 * (5 * La) ** (1 / 3)
    FL = ne.evaluate("0.2 * k ** 4 * (5 * La) + 0.1 * (1 - k ** 4) ** 2 * (5 * La) ** (1 / 3)")
    FL_rep = np.stack([FL, FL, FL], axis=2)
    detail_a = ne.evaluate("detail ** ((FL_rep + 0.8) ** 0.25)")
    return detail_a


def main():
    # read mat
    # Input of the iCAM06 model: XYZ, absolute color space
    XYZ = np.load('xyz.npy').astype(np.float32)

    # Image decomposition
    base_layer, detail_layer = bilateral_filter(XYZ)
    # base_layer = XYZ

    # Chromatic adaptation
    white = blur(XYZ, 2)
    # XYZ_adapt = iCAM06_CAT(base_layer, white)
    XYZ_adapt = img_CAT02_to_D65(base_layer, white, surround="average")

    # Tone compression
    white = blur(XYZ, 3)
    # XYZ_tc = iCAM06_TC(XYZ_adapt, white, 0.7)
    XYZ_tc = img_TC(XYZ_adapt, white, 0.7)

    # Image attribute adjustments
    XYZ_d = XYZ_tc * LocalContrast(detail_layer, base_layer)
    # XYZ_d = XYZ_tc
    # XYZ_p = iCAM06_IPT(XYZ_d, base_layer, 1.0)
    XYZ_p = IPT(XYZ_d)

    # Convert XYZ to RGB
    
    RGB_p = XYZ_to_sRGB(XYZ_p)
    '''
    # Display the image
    
    plt.imshow(RGB_p)
    plt.show()
    '''
    # Convert RGB from RGB to BGR for OpenCV
    
    RGB_p_bgr = cv2.cvtColor((RGB_p * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # Save the image
    cv2.imwrite('output.jpg', RGB_p_bgr)
    
    
if __name__ == "__main__":
    main()
