import cv2
import numpy as np
import matplotlib.pyplot as plt


'''低频信息主要表示的是一张图的总体样貌，一般低频系数的值也比较大。
而高频信息主要表示的是图像中人物或物体的细节，一般高频系数的数量较多'''
def image_dct_transform(img):
    img_f = np.float32(img)  # dct输入必须为单通道浮点型
    height, width = img.shape

    img_dct = cv2.dct(img_f)  # DCT 结果是实数数组，不是DFT的复数数组
    dct_out = np.log(np.abs(img_dct) + 1)  # 画图用

    dct_lowpass = np.zeros_like(img_dct)
    dct_lowpass[:height//5, :width//5] = img_dct[:height//5, :width//5]  # 保留左上角50x50区域的低频信号
    dct_out_lowpass = np.log(np.abs(dct_lowpass) + 1)   # 图像的大部分信息都在低频信息zhong

    dct_highpass = np.zeros_like(img_dct)
    dct_highpass[height//10:, width//10:] = img_dct[height//10:, width//10:]  # 保留右下角的高频信号
    dct_out_highpass = np.log(np.abs(dct_highpass) + 1)

    # 对滤波后的DCT图像进行逆变换
    img_idct_lowpass = cv2.idct(dct_lowpass)
    img_idct_highpass = cv2.idct(dct_highpass)

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title('DCT Image')
    plt.imshow(dct_out, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title('Loswpass Image')
    plt.imshow(img_idct_lowpass, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title('Loswpass DCT Image')
    plt.imshow(dct_out_lowpass, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Highpass Image')
    plt.imshow(img_idct_highpass, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.title('Highpass DCT Image')
    plt.imshow(dct_out_highpass, cmap='gray')

    plt.tight_layout()
    plt.show()

'''数字图像这种离散的信号，频率大小表示信号变化的剧烈程度或者说是信号变化的快慢。
频率越大，变化越剧烈，频率越小，信号越平缓。 对应到图像中，高频信号往往是图像中的边缘信号和噪声信号，
而低频信号包含图像变化频繁的图像轮廓及背景等信号。'''
def image_dft_transform(img):
    img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)    # 对一维或二维的实数数组进行正向变换
    img_dft_shift = np.fft.fftshift(img_dft)    # 中心化
    # 计算频谱图（取绝对值并对数处理）
    magnitude_spectrum = 20 * np.log(cv2.magnitude(img_dft_shift[:, :, 0], img_dft_shift[:, :, 1]) + 1)

    # 掩膜 创建低频滤波器，保留中心区域的低频信息
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2  # 找到中心点
    mask_lowpass = np.zeros((rows, cols, 2), np.uint8)  # 创建一个形如img的全0矩阵
    mask_lowpass[crow-30:crow+30, :] = 1  # 60x60 的低频保留区域系数为1
    mask_lowpass[:, ccol-30:ccol+30] = 1  # 60x60 的低频保留区域系数为1

    # 应用低频滤波器
    fshift_lowpass = img_dft_shift * mask_lowpass
    lowpass_spectrum = np.log(cv2.magnitude(fshift_lowpass[:, :, 0], fshift_lowpass[:, :, 1]) + 1)
    img_idft_lowpass = cv2.idft(np.fft.ifftshift(fshift_lowpass))
    img_idft_lowpass = cv2.magnitude(img_idft_lowpass[:, :, 0], img_idft_lowpass[:, :, 1])
    

    # 创建高频滤波器，保留中心外的高频信息
    mask_highpass = np.ones((rows, cols, 2), np.uint8)   # 创建一个形如img的全0矩阵
    mask_highpass[crow-30:crow+30, :] = 0  # 中心60x60区域置零
    mask_highpass[:, ccol-30:ccol+30] = 0  # 中心60x60区域置零
    # 应用高频滤波器
    fshift_highpass = img_dft_shift * mask_highpass
    highpass_spectrum = np.log(cv2.magnitude(fshift_highpass[:, :, 0], fshift_highpass[:, :, 1]) + 1)
    img_idft_highpass = cv2.idft(np.fft.ifftshift(fshift_highpass))
    img_idft_highpass = cv2.magnitude(img_idft_highpass[:, :, 0], img_idft_highpass[:, :, 1])

    # 显示结果
    plt.figure()

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')

    # DFT 频谱图
    plt.subplot(2, 3, 4)
    plt.title('DFT Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')

    # 低频滤波后的频谱图
    plt.subplot(2, 3, 5)
    plt.title('Lowpass DFT Spectrum')
    plt.imshow(lowpass_spectrum, cmap='gray')

    # 低频滤波后图像
    plt.subplot(2, 3, 2)
    plt.title('Lowpass Filtered')
    plt.imshow(img_idft_lowpass, cmap='gray')

    # 高频滤波后的频谱图
    plt.subplot(2, 3, 6)
    plt.title('Highpass DFT Spectrum')
    plt.imshow(highpass_spectrum, cmap='gray')

    # 高频滤波后图像
    plt.subplot(2, 3, 3)
    plt.title('Highpass Filtered')
    plt.imshow(img_idft_highpass, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img = cv2.imread("../image/cameraman.tif", cv2.IMREAD_GRAYSCALE)    # 读取灰度图像
    image_dct_transform(img)
    # image_dft_transform(img)
