import numpy as np


M_H_D65 = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)

M_IPT = np.array(
    [
        [0.4000, 0.4000, 0.2000],
        [4.4550, -4.8510, 0.3960],
        [0.8056, 0.3572, -1.1628],
    ]
)

def IPT(XYZ):
    XYZ_reshape = XYZ.reshape((-1, 3))
    LMS = XYZ_reshape @ M_H_D65.T
    
    LMS_nonlinear = np.sign(LMS) * np.abs(LMS) ** 0.43
    
    IPT = LMS_nonlinear @ M_IPT.T
    IPT = IPT.reshape(XYZ.shape)
    
    L_A = 0.2 * XYZ[..., 1]
    k = 1.0 / (5 * L_A + 1)
    F_L = 0.2 * k ** 4 * (5 * L_A) + 0.1 * (1 - k ** 4) ** 2 * (5 * L_A) ** (1 / 3)
    
    C = np.sqrt(IPT[..., 1] ** 2 + IPT[..., 2] ** 2)
    
    adjustment = (F_L + 1) ** 0.2 * ((1.29 * C ** 2 - 0.27 * C + 0.42) / (C ** 2 - 0.31 * C + 0.42))
    
    IPT[..., 1] = IPT[..., 1] * adjustment
    IPT[..., 2] = IPT[..., 2] * adjustment
    
    # IPT[..., 0] = IPT[..., 0] ** 1.0
    
    IPT_reshape = IPT.reshape((-1, 3))
    LMS_nonlinear = IPT_reshape @ np.linalg.inv(M_IPT).T
    LMS = np.sign(LMS_nonlinear) * np.abs(LMS_nonlinear) ** (1 / 0.43)
    XYZ_reshape = LMS @ np.linalg.inv(M_H_D65).T
    XYZ = XYZ_reshape.reshape(XYZ.shape)
    
    return XYZ
    
    