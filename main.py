import sys
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
from UI.Ui_window_ui import Ui_MainWindow  
from src.image_transform import image_dct_transform, image_dft_transform
from src.image_enhance import image_sharpen, color_histogram_equalization
from src.image_restore import iterative_blind_deconv
from src.image_encode import *


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.run_code)

    def run_code(self):
        option = self.comboBox.currentText()
        if option == "DCT":
            self.textBrowser.clear()
            img = cv2.imread("./image/cameraman.tif", cv2.IMREAD_GRAYSCALE)
            image_dct_transform(img)
        elif option == "DFT":
            self.textBrowser.clear()
            img = cv2.imread("./image/cameraman.tif", cv2.IMREAD_GRAYSCALE)
            image_dft_transform(img)
        elif option == "直方图均衡化":
            self.textBrowser.clear()
            image = cv2.imread('./image/underexposed_43.jpg')  # 替换为你的图像路径
            color_histogram_equalization(image)
        elif option == "图像锐化":
            self.textBrowser.clear()
            img = cv2.imread("./image/house.tif", cv2.IMREAD_GRAYSCALE)
            image_sharpen(img)
        elif option == "IBD":
            self.textBrowser.clear()
            self.textBrowser.setText("盲目反卷积运行中")
            iterative_blind_deconv()
        elif option == "Huffman编码":
            img = cv2.imread("./image/pirate.tif", cv2.IMREAD_GRAYSCALE) # 读取灰度图
            '''第一步, 获取编码表'''
            original_bits = img.size * 8
            entropy, histogram = calculate_entropy(img)
            huffman_tree = generate_huffman_tree(histogram)  # 生成霍夫曼树
            codebook = generate_huffman_codes(huffman_tree)  # 生成编码表
            average_code_length = calculate_average_code_length(histogram, codebook)
            compressed_bits = sum(len(codebook[symbol]) * frequency for symbol, frequency in histogram.items())
            save_huffman_codebook_to_csv(codebook, "./output/pirate_codebook.csv")
            self.textBrowser.insertPlainText(f"源图像的熵: {entropy:.4f}\n")
            self.textBrowser.insertPlainText(f"编码后的平均码字长度: {average_code_length:.4f}\n")
            self.textBrowser.insertPlainText(f"大小为原来的{compressed_bits/original_bits*100:.2f}%\n")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(img, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title("Huffman Codes (partial)")
            plt.bar(range(len(codebook)), [len(code) for code in codebook.values()])
            plt.xlabel("Symbol")
            plt.ylabel("Code Length")
            plt.show()
        elif option == "Huffman重建":
            img = cv2.imread("./image/pirate.tif", cv2.IMREAD_GRAYSCALE)
            codebook = {}
            with open("./output/pirate_codebook.csv", mode="r") as file:
                reader = csv.reader(file)
                next(reader)                    # 跳过表头
                for row in reader:
                    symbol = int(row[0])       # 转换符号为整数（像素值）
                    code = row[1]              # 霍夫曼编码保持为字符串
                    codebook[symbol] = code     # codebook字典
            reverse_codebook = {code: symbol for symbol, code in codebook.items()} # 反转，从码字映射到灰度值
            appendix_len, encoded_img = img_encode_to_string(img, codebook, "./output/pirate.bin")
            '''第三步 从bin解压缩'''
            encoded_img = "".join(codebook[pixel] for pixel in img.flatten())
            print(len(encoded_img))
            restored_img = huffman_decode("./output/pirate.bin", reverse_codebook, img.shape, appendix_len)
            plt.figure()
            plt.subplot(111)
            plt.imshow(restored_img, cmap="gray")
            plt.imsave("./output/pirate_decoded.png", restored_img, cmap="gray")
            plt.show()

            img = Image.open("./output/pirate_decoded.png").convert('L')
            img.save('./output/pirate_decoded.tif') # 转换后的进行保存


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
