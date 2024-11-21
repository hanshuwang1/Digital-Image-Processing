import cv2
import numpy as np
import matplotlib.pyplot as plt


# TODO 写成类，降低耦合，留出接口
def histogram_equaliaztion(img):
    equalized_img = cv2.equalizeHist(img)

    plt.figure()

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')

    # 原始图像的直方图
    plt.subplot(2, 2, 3)
    plt.title('Histogram of Original Image')
    plt.hist(img.ravel(), 256, [0, 256])    # 0-255 256个条柱 range[0, 256]
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # 均衡化后的图像
    plt.subplot(2, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_img, cmap='gray')

    # 均衡化后图像的直方图
    plt.subplot(2, 2, 4)
    plt.title('Histogram of Equalized Image')
    plt.hist(equalized_img.ravel(), 256, [0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def image_sharpen(img):
    # img = cv2.GaussianBlur(img, (5, 5), 1)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.blur(img, (5,5))
    img = cv2.medianBlur(img, 5)
    roberts_kernel_l = np.array([[-1, 0],
                                 [0, 1]], dtype=np.float32)
    roberts_kernel_r = np.array([[0, -1],
                                 [1, 0]], dtype=np.float32)
    laplace_kernel = np.array([[0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]], dtype=np.float32)
    sobel_kernel_h = np.array([[-1, 0, 1],
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=np.float32)
    sobel_kernel_v = np.array([[-1, -2, -1],
                                [0, 0, 0], 
                                [1, 2, 1]], dtype=np.float32)                           
    img_roberts_l = cv2.filter2D(img, -1, roberts_kernel_l)   # -1表示前后输出深度相同
    img_roberts_r = cv2.filter2D(img, -1, roberts_kernel_r)
    img_roberts = cv2.addWeighted(img_roberts_l, 0.5, img_roberts_r, 0.5, 0)
    img_roberts = img_roberts.astype("uint8", copy=False)
    _, roberts_binary_image = cv2.threshold(img_roberts, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img_laplace = cv2.filter2D(img, -1, laplace_kernel)
    img_laplace = img_laplace.astype("uint8", copy=False)
    _, laplacian_binary_image = cv2.threshold(img_laplace, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img_sobel_h = cv2.filter2D(img, -1, sobel_kernel_h)
    img_sobel_v = cv2.filter2D(img, -1, sobel_kernel_v)
    img_sobel = cv2.addWeighted(img_sobel_h, 0.5, img_sobel_v, 0.5, 0)
    img_sobel = img_sobel.astype("uint8", copy=False)
    _, sobel_binary_image = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('roberts')
    plt.imshow(roberts_binary_image, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('laplace')
    plt.imshow(laplacian_binary_image, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('sobel')
    plt.imshow(sobel_binary_image, cmap='gray')

    plt.tight_layout()
    plt.show()

# 绘制直方图函数
def plot_histogram(ax, channel_data, channel_labels, title):
    colors = ['b', 'g', 'r']  # 通道颜色
    for data, label, color in zip(channel_data, channel_labels, colors):
        ax.hist(data.ravel(), bins=256, color=color, alpha=0.5, label=label)
    ax.set_title(title)
    ax.legend()

def hist_equalizton_rgb(image):
    b, g, r = cv2.split(image) # 通道的顺序是BGR
    
    # 对每个通道分别进行直方图均衡化
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    
    # 合并均衡化后的通道
    equalized_image = cv2.merge((b_eq, g_eq, r_eq))

    # 转换 BGR 到 RGB 格式，以便 Matplotlib 显示
    equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
    return equalized_image_rgb, (b_eq, g_eq, r_eq)

    
def hist_equalizton_hsv(image:np.ndarray):
    # 转成HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 对 Y 通道（亮度）进行直方图均衡化
    hsv_image[:, :, 0] = cv2.equalizeHist(hsv_image[:, :, 0])
    equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_YUV2RGB)
    b, g, r = cv2.split(equalized_image)
    return equalized_image, (b, g, r)

def color_histogram_equalization(image):
    
    b_orig, g_orig, r_orig = cv2.split(image)

    equalized_image_rgb,  (b_eq_rgb, g_eq_rgb, r_eq_rgb)= hist_equalizton_rgb(image)
    equalized_image_hsv, (b_eq_hsv, g_eq_hsv, r_eq_hsv) = hist_equalizton_hsv(image)
    # 创建图形窗口
    plt.figure(figsize=(12, 8))

    # 显示原图
    plt.subplot(321)
    plt.title("Original Image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')

    # 显示原图直方图
    plt.subplot(322)
    plot_histogram(plt.gca(), [b_orig, g_orig, r_orig], ['Blue', 'Green', 'Red'], "Original Histogram")

    # 显示处理后图像
    plt.subplot(323)
    plt.title("RGB Equalized Image")
    plt.imshow(equalized_image_rgb)
    plt.axis('off')

    # 显示处理后直方图
    plt.subplot(324)
    plot_histogram(plt.gca(), [b_eq_rgb, g_eq_rgb, r_eq_rgb], ['Blue', 'Green', 'Red'], "Equalized Histogram")

    # 显示处理后图像
    plt.subplot(325)
    plt.title("HSV Equalized Image")
    plt.imshow(equalized_image_hsv)
    plt.axis('off')

    # 显示处理后直方图
    plt.subplot(326)
    plot_histogram(plt.gca(), [b_eq_hsv, g_eq_hsv, r_eq_hsv], ['Blue', 'Green', 'Red'], "Equalized Histogram")

    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img = cv2.imread("../image/house.tif", 0)
    # histogram_equaliaztion(img)
    image_sharpen(img)

    # 加载曝光不足的彩色图像
    # image = cv2.imread('../image/underexposed_43.jpg')  # 替换为你的图像路径
    # color_histogram_equalization(image)
