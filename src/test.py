import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from image_restore import blind_deconvo

def gaussian_psf(size, sigma=1):
    gaussian_kernel = cv2.getGaussianKernel(size, sigma)
    gaussian_kernel_2d = gaussian_kernel.dot(gaussian_kernel.T)
    return gaussian_kernel_2d


def deconvblind(blurred, psf_init, iterations=10, clip=True):
    latent = np.full_like(blurred, 0.5)  # 初始化估计的图像
    psf = psf_init.copy()
    psf_size = psf.shape  # 保存 PSF 的固定大小

    for i in range(iterations):
        # Step 1: 更新清晰图像
        estimated_blur = convolve2d(latent, psf, mode='same', boundary='symm')
        relative_blur = blurred / (estimated_blur + 1e-8)
        latent = latent * convolve2d(relative_blur, np.flip(psf), mode='same', boundary='symm')

        # Step 2: 更新 PSF
        estimated_blur = convolve2d(latent, psf, mode='same', boundary='symm')
        relative_blur = blurred / (estimated_blur + 1e-8)
        psf_update = convolve2d(np.flip(latent), relative_blur, mode='same', boundary='symm')
        
        # 裁剪 PSF 更新结果以保持固定大小
        center = tuple(s // 2 for s in psf_update.shape)
        half_size = tuple(s // 2 for s in psf_size)
        psf *= psf_update[
            center[0] - half_size[0]:center[0] + half_size[0] + 1,
            center[1] - half_size[1]:center[1] + half_size[1] + 1,
        ]

        psf = np.clip(psf, 0, None)  # 保证 PSF 非负
        psf = psf / (np.sum(psf) + 1e-8)  # 归一化 PSF
        
        # 可选：裁剪图像值域
        if clip:
            latent = np.clip(latent, 0, 1)

    return latent, psf

# 测试代码
if __name__ == "__main__":

    # 模拟模糊图像
    original = np.zeros((100, 100))
    original[40:60, :] = 1
    original[:, 40:60] = 1
    psf_true = gaussian_psf(15, 15)  # 真正的 PSF
    blurred = convolve2d(original, psf_true, mode='same', boundary='symm')

    # 初始化 PSF
    psf_init = np.ones((15, 15)) / 225

    # 盲去卷积
    latent, psf_estimated = deconvblind(blurred, psf_init, iterations=10)
    # latent, psf_estimated = blind_deconvo(blurred, psf_init, 20)
    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Blurred Image")
    plt.imshow(blurred, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Restored Image")
    plt.imshow(latent, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Estimated PSF")
    plt.imshow(psf_estimated, cmap='gray')
    plt.show()
