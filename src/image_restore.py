import numpy as np
import cv2
from scipy.signal import convolve2d, fftconvolve 
import matplotlib.pyplot as plt

def gaussian_psf(size, sigma=1):
    gaussian_kernel = cv2.getGaussianKernel(size, sigma)
    gaussian_kernel_2d = gaussian_kernel.dot(gaussian_kernel.T)
    return gaussian_kernel_2d

def lucy_richardson(observed: np.ndarray, psf: np.ndarray, num_iter=10):
    '''an iterative procedure for recovering an underlying image 
    that has been blurred by a KONWN psf.'''
    observed = observed.astype(np.float64, copy=False)
    psf = psf.astype(np.float64, copy=False)

    # 初始化估计图像为观测图像
    img_deconv = np.full_like(observed, 0.5, dtype=np.float64)  # 初始图像估计
    
    for _ in range(num_iter):
        conv = convolve2d(img_deconv, psf, mode='same')
        relative_blur = observed / (conv + 1e-6)  # 防止除以零
        img_deconv *= convolve2d(relative_blur, np.flip(psf), mode='same')
        
    return img_deconv.astype(np.uint8)


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


def get_image_part(image: np.ndarray, start_x=250, start_y=230, size=100, border=5):
    image = image.copy()
    part = image[start_y:start_y+size, start_x:start_x+size]
    image[start_y-border:start_y, start_x:start_x+size] = 255
    image[start_y+size:start_y+size+border, start_x:start_x+size] = 255
    image[start_y:start_y+size, start_x-border:start_x] = 255
    image[start_y:start_y+size, start_x+size:start_x+size+border] = 255
    add_border_image = image
    return add_border_image, part 

def iterative_blind_deconv():
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
if __name__ == "__main__":
    # 构建示例模糊图像和PSF
    original_image = cv2.imread("../image/cameraman.tif", cv2.IMREAD_GRAYSCALE) # 假设原始图像为随机图像
    original_image_show, part_original = get_image_part(original_image)
    psf = gaussian_psf(15, 15)
    int_psf = np.ones(psf.shape) / 225
    # print(type(est_psf))

    # 模糊图像
    blur_image = cv2.filter2D(original_image, -1, psf)
    blur_image_show, part_blurred = get_image_part(blur_image)

    # 使用Lucy-Richardson算法复原图像
    # restored_image = lucy_richardson(blur_image, psf, num_iter=20)
    # restored_image, part_restored = get_image_part(restored_image)

    # 盲目反卷积
    restored_image, est_psf = deconvblind(blur_image, int_psf, 10)
    estored_image, part_restored = get_image_part(restored_image)

    # 可视化结果
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image_show, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("Blurred Image ")
    plt.imshow(blur_image_show, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Restored Image")
    plt.imshow(restored_image, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("part_original")
    plt.imshow(part_original, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("part_blur")
    plt.imshow(part_blurred, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("part_restored")
    plt.imshow(part_restored, cmap='gray')

    plt.tight_layout()
    plt.show()
