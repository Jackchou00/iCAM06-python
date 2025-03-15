# Python Implementation of the Image Color Appearance Model (iCAM)

English | [简体中文](ReadMe_zh.md)

## Introduction

Color appearance models, as an extension of colorimetry, have significant applications in image processing, particularly in tone mapping for high dynamic range (HDR) imaging.

The iCAM framework, proposed by Fairchild and colleagues, adapts color appearance models to image processing. The iCAM06 version incorporates state-of-the-art colorimetric methods and models from its time, such as CAT02, CAM02, and IPT. This framework has since been refined and extended, with derivatives like TMOz.

This project provides a Python implementation of the iCAM framework, based on the original MATLAB code for iCAM06.

## References

[1] J. Kuang, G. M. Johnson, and M. D. Fairchild, “iCAM06: A refined image appearance model for HDR image rendering,” Journal of Visual Communication and Image Representation, vol. 18, no. 5, pp. 406–414, Oct. 2007, doi: 10.1016/j.jvcir.2007.06.003.

[2] I. Mehmood, M. Zhou, M. U. Khan, and M. Luo, “CIECAM16-based Tone Mapping of High Dynamic Range Images,” Color and Imaging Conference, vol. 31, pp. 102–107, Nov. 2023, doi: 10.2352/CIC.2023.31.1.20.

## Files

The project is currently undergoing structural improvements to enhance modularization and readability, aiming for a more organized and maintainable codebase.

The existing code is process-oriented, including:

- `iCAM06.py`: The main implementation of the iCAM06 algorithm.
- `main.py`: An example of how to use the iCAM06 algorithm.
- `fastbiliateral_blur.py`: A fast implementation of the bilateral filter and Gaussian blur.
- `xyz.npy`: XYZ example data from Fairchild's original code. (resampled to 1/4 size)
- `output.jpg`: The output of the example code.
