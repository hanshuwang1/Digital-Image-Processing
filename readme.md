# 须知

## 本工程是数字图像处理编程作业

## requirement

- graphviz                  0.20.3  查看huffman编码二叉树
- matplotlib                3.9.2   画图和显示
- numpy                     2.0.2
- opencv-python             4.10.0.84
- pillow                    11.0.0  图片格式转换
- PyQt5                     5.15.9  UI
- pyqt5-tools               5.15.9.3.3
- scipy                     1.13.1  卷积

## 任务

- [x]  DFT和DCT实现
- [x]  直方图均衡化实现
- [x]  锐化实现
- [ ]  IBD算法使用了迭代公式，没用真实图片原因是运算太慢。
- [x]  Huffman编码解码实现，
- [x]  UI界面 PyQt5

## 文件夹解释

- image 存放原始图像
- output 存放处理后图像
- src 算法的具体实现
- UI 交互界面
- venv-python39-dip 装了python3.9.20版本的虚拟环境
- main.py 主程序，运行打开UI界面

## 运行

- 通过vscode打开文件夹，选择python解释器为工程中的venv-python39-dip
- 打开main.py 文件，运行
- 点击运行按钮后跳出图片显示界面，输出一些信息，关闭图片显示界面后再运行其他算法
- 界面文本框输出信息

## 参考

- 测试图片来源 [https://github.com/Whisper329/image-processing.git]
[https://github.com/201712530136/UUMImages.git]
