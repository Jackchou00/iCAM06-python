# 图像色貌模型（iCAM）的 Python 实现

[English](ReadMe.md) | 简体中文

## 简介

色貌模型作为色度学的延伸，在图像处理领域也有重要的应用，尤其是高动态范围图像的色调映射方面。

iCAM 是 Fairchild 等人提出的一种框架，将色貌模型应用于图像处理。其中 iCAM06 版本用到了当时最先进的色度学方法和模型，比如 CAT02，CAM02，IPT 等。这种框架后来也有改进和衍生，比如 TMOz。

这个项目是 iCAM 的 Python 实现，参考了 iCAM06 的原始代码（MATLAB）。

## 参考文献

[1] J. Kuang, G. M. Johnson, and M. D. Fairchild, “iCAM06: A refined image appearance model for HDR image rendering,” Journal of Visual Communication and Image Representation, vol. 18, no. 5, pp. 406–414, Oct. 2007, doi: 10.1016/j.jvcir.2007.06.003.

[2] I. Mehmood, M. Zhou, M. U. Khan, and M. Luo, “CIECAM16-based Tone Mapping of High Dynamic Range Images,” Color and Imaging Conference, vol. 31, pp. 102–107, Nov. 2023, doi: 10.2352/CIC.2023.31.1.20.

## 文件

目前，正在改进项目整体的文件结构，以实现模块化和更好的可读性。

现有的文件是面向过程的，包括：

- `iCAM06.py`: iCAM06算法的主要实现。
- `main.py`: 使用iCAM06算法的示例。
- `fastbiliateral_blur.py`: 双边滤波器和高斯模糊的快速实现。
- `xyz.npy`: 来自Fairchild原始代码的XYZ示例数据。（重采样为1/4大小）
- `output.jpg`: 示例代码的输出。
