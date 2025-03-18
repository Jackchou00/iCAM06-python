import numpy as np
import numexpr as ne
from spatial_process.fastbiliateral_blur import bilateral_filter, blur

# from iCAM06 import iCAM06_CAT, iCAM06_TC, iCAM06_IPT
from chromatic_adaptation import img_modified_CAT02_to_D65, img_vK20_to_D65
from colour_space_conversion import XYZ_to_sRGB, XYZ_to_IPT, IPT_to_XYZ, XYZ_to_sUCS_Iab, sUCS_Iab_to_XYZ, XYZ_to_P3_RGB
from tone_compression.TC import img_TC
from colour_space_conversion.IPT_adjust import IPT_adjust
from colour_space_conversion.sUCS_adjust import sUCS_adjust
import cv2
from PIL import Image


def LocalContrast(detail, base):
    La = 0.2 * base[:, :, 1]
    k = 1.0 / (5 * La + 1)
    # FL = 0.2 * k ** 4 * (5 * La) + 0.1 * (1 - k ** 4) ** 2 * (5 * La) ** (1 / 3)
    FL = ne.evaluate(
        "0.2 * k ** 4 * (5 * La) + 0.1 * (1 - k ** 4) ** 2 * (5 * La) ** (1 / 3)"
    )
    FL_rep = np.stack([FL, FL, FL], axis=2)
    detail_a = ne.evaluate("detail ** ((FL_rep + 0.8) ** 0.25)")
    return detail_a


def main():
    # read mat
    # Input of the iCAM06 model: XYZ, absolute color space
    XYZ = np.load("example/xyz.npy").astype(np.float32)
    print(np.max(XYZ), np.min(XYZ))

    # Image decomposition
    base_layer, detail_layer = bilateral_filter(XYZ)
    print(np.max(base_layer), np.min(base_layer))
    # base_layer = XYZ

    # Chromatic adaptation
    white = blur(XYZ, 2)
    # XYZ_adapt = iCAM06_CAT(base_layer, white)
    # XYZ_adapt = img_modified_CAT02_to_D65(base_layer, white, surround="average")
    XYZ_adapt = img_vK20_to_D65(base_layer, white, surround="average")
    print(np.max(XYZ_adapt), np.min(XYZ_adapt))

    # Tone compression
    white = blur(XYZ, 3)
    # XYZ_tc = iCAM06_TC(XYZ_adapt, white, 0.7)
    XYZ_tc = img_TC(XYZ_adapt, white, 0.7)
    print(np.max(XYZ_tc), np.min(XYZ_tc))

    # Image attribute adjustments
    XYZ_d = XYZ_tc * LocalContrast(detail_layer, base_layer)
    print(np.max(XYZ_d), np.min(XYZ_d))
    # XYZ_d = XYZ_tc
    # XYZ_p = iCAM06_IPT(XYZ_d, base_layer, 1.0)
    # IPT = XYZ_to_IPT(XYZ_d)
    # IPT_adjusted = IPT_adjust(IPT, XYZ_d)
    # XYZ_p = IPT_to_XYZ(IPT_adjusted)
    
    sUCS = XYZ_to_sUCS_Iab(XYZ_d)
    sUCS_adjusted = sUCS_adjust(sUCS, XYZ_d)
    XYZ_p = sUCS_Iab_to_XYZ(sUCS_adjusted)
    
    print(np.max(XYZ_p), np.min(XYZ_p))

    # Convert XYZ to RGB

    # RGB_p = XYZ_to_sRGB(XYZ_p)
    RGB_p = XYZ_to_P3_RGB(XYZ_p)
    """
    # Display the image
    
    plt.imshow(RGB_p)
    plt.show()
    """
    # Convert RGB from RGB to BGR for OpenCV

    # Convert RGB to uint8 range for saving
    RGB_p_uint8 = (RGB_p * 255).clip(0, 255).astype(np.uint8)
    
    # Create PIL Image and save
    img = Image.fromarray(RGB_p_uint8)
    with open("ICC/Display P3.icc", "rb") as icc_file:
        icc_profile = icc_file.read()
    img.save("example/output_sUCS_P3.jpg", icc_profile=icc_profile)
    print("Image saved as example/output_sUCS_P3.jpg")


if __name__ == "__main__":
    main()
