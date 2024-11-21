import csv
from collections import Counter
from heapq import heapify, heappop, heappush
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from graphviz import Digraph

def calculate_entropy(image):
    histogram = Counter(image.flatten())    # 获取像素值的频率分布
    total_pixels = image.size
    entropy = -sum((count / total_pixels) * np.log2(count / total_pixels) for count in histogram.values())
    return entropy, histogram

# 定义霍夫曼节点类
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol    # 不同灰度级的码字
        self.freq = freq        # 概率
        self.left = None        # 左子节点
        self.right = None       # 右子节点
    
    def __lt__(self, other):    # 比大小
        return self.freq < other.freq

# 构建霍夫曼树
def generate_huffman_tree(histogram):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in histogram.items()]    # 对每一个灰度级生成一个
    heapify(heap)   #生成大根堆 从小到大排序
    
    while len(heap) > 1:
        node1 = heappop(heap)   # 弹出最小概率的灰度值子节点
        node2 = heappop(heap)   # 弹出第二小概率的灰度值子节点
        merged = HuffmanNode(None, node1.freq + node2.freq) #  创建新节点，其频率为两个最小节点的频率之和 
        merged.left = node1     # 将最小节点设为左子节点
        merged.right = node2    # 将次小节点设为右子节点
        heappush(heap, merged)  # 将合并后的节点重新放入堆中-大根堆
        
    return heap[0] if heap else None    # 剩下的为根节点

# 递归
def generate_huffman_codes(node, prefix="", codebook={}):
    # 如果节点为空，则返回空编码表
    if node is None:
        return
    # 如果节点是叶子节点，将其编码添加到编码表
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    # 递归生成左子节点的编码，编码前缀加 '0'
    generate_huffman_codes(node.left, prefix + "0", codebook)
    # 递归生成右子节点的编码，编码前缀加 '1'
    generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook


# 计算平均码字长度
def calculate_average_code_length(histogram: dict, codebook):
    # 计算图像的总符号数量
    total_symbols = sum(histogram.values())
    # 计算加权平均码字长度
    average_code_length = sum((histogram[symbol] / total_symbols) * len(codebook[symbol]) for symbol in codebook)
    return average_code_length


def save_huffman_codebook_to_csv(codebook, filename="huffman_codes.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol", "Huffman Code"])  # 写入表头
        for symbol, code in codebook.items():
            writer.writerow([symbol, code])  # 写入符号和对应的霍夫曼编码
    file.close()


def img_encode_to_string(img, codebook, savepath="encoded_image.bin"):
    encoded_img = "".join(codebook[pixel] for pixel in img.flatten()) # 遍历每个像素值,查灰度值得到码字，拼接字符串
    # 将二进制字符串按每 8 位分割并转换为字节
    byte_array = bytearray()
    for i in range(0, len(encoded_img), 8):
        byte_chunk = encoded_img[i:i+8]
        # 如果最后一个字节不足 8 位，用 '0' 补齐
        if len(byte_chunk) < 8:
            temp = 8 - len(byte_chunk)
            byte_chunk = byte_chunk.ljust(8, '0')
        # 将每 8 位的二进制字符串转换为一个字节，并添加到 bytearray
        byte_array.append(int(byte_chunk, 2))

    # 3. 保存为二进制文件
    with open(savepath, "wb") as f:  #b表示二进制文件
        f.write(byte_array)
    print(f"补了{temp}个0")
    return temp, encoded_img


def huffman_decode(img_encoded_path, reverse_codebook:dict, shape, appendix_len):
    decoded_pixels = []
    current_code = ""
    with open(img_encoded_path, "rb") as f:
        byte_data = f.read()  # 读取所有字节
    img_encoded_string = ''.join(f'{byte:08b}' for byte in byte_data)   # 将每个字节转换为二进制字符串（8位）并拼接
    # TODO: 先从较长的码字开始匹配
    img_encoded_string = img_encoded_string[:-appendix_len]
    for bit in img_encoded_string:
        current_code += bit  # 累加当前读取的位  唯一可译码
        if current_code in reverse_codebook: # 找到一个完整的霍夫曼编码对应的像素值
            decoded_pixels.append(reverse_codebook[current_code])
            current_code = ""  # 重置当前编码，继续处理剩余编码串
    print(f"解码出{len(decoded_pixels)}个像素")
    decoded_image = np.array(decoded_pixels, dtype=np.uint8).reshape(shape)
    return decoded_image


def visualize_huffman_tree(root, codebook):

    dot = Digraph(comment='Huffman Tree', format='png')
    
    def add_nodes_edges(node, parent_id=None, edge_label=None):
        if not node:
            return

        node_id = str(id(node))
        # 节点标签
        label = f"{codebook[node.symbol] if node.symbol != None else ''} ({node.freq}){node.symbol or ''}"
        dot.node(node_id, label)
        
        # 添加边
        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)

        # 递归添加左右子树
        add_nodes_edges(node.left, node_id, "0")
        add_nodes_edges(node.right, node_id, "1")
    
    add_nodes_edges(root)
    return dot


if __name__ == '__main__':
    img = cv2.imread("../image/pirate.tif", cv2.IMREAD_GRAYSCALE) # 读取灰度图
    '''第一步, 获取编码表'''
    # print(type(img))
    # img = np.array([[255, 254, 254, 254], 
    #                 [255, 255, 253, 252], 
    #                 [255, 255, 255, 253], 
    #                 [255, 254, 253, 252]])
    # original_bits = img.size * 8
    # print(original_bits)
    entropy, histogram = calculate_entropy(img)
    huffman_tree = generate_huffman_tree(histogram)  # 生成霍夫曼树
    codebook = generate_huffman_codes(huffman_tree)
    huffman_tree_graph = visualize_huffman_tree(huffman_tree, codebook)
    huffman_tree_graph.render('../output/huffman_tree', view=True)  # 保存并显示图像


    # codebook = generate_huffman_codes(huffman_tree)  # 生成编码表
    # average_code_length = calculate_average_code_length(histogram, codebook)
    # compressed_bits = sum(len(codebook[symbol]) * frequency for symbol, frequency in histogram.items())
    # print(compressed_bits)
    # save_huffman_codebook_to_csv(codebook, "../output/pirate_codebook.csv")
    # print(f"源图像的熵: {entropy:.4f}")
    # print(f"编码后的平均码字长度: {average_code_length:.4f}")
    # print(f"大小为原来的{compressed_bits/original_bits*100:.2f}%")


    # # 可视化编码结果
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("Huffman Codes (partial)")
    # plt.bar(range(len(codebook)), [len(code) for code in codebook.values()])
    # plt.xlabel("Symbol")
    # plt.ylabel("Code Length")
    # plt.show()
    

    # codebook = {}
    # with open("../output/pirate_codebook.csv", mode="r") as file:
    #     reader = csv.reader(file)
    #     next(reader)                    # 跳过表头
    #     for row in reader:
    #         symbol = int(row[0])       # 转换符号为整数（像素值）
    #         code = row[1]              # 霍夫曼编码保持为字符串
    #         codebook[symbol] = code     # codebook字典
    # reverse_codebook = {code: symbol for symbol, code in codebook.items()} # 反转，从码字映射到灰度值
    '''第二步  利用编码表压缩图片到bin格式'''
    # appendix_len, encoded_img = img_encode_to_string(img, codebook, "../output/pirate.bin")
    '''第三步 从bin解压缩'''
    # encoded_img = "".join(codebook[pixel] for pixel in img.flatten())
    # print(len(encoded_img))
    # restored_img = huffman_decode("../output/pirate.bin", reverse_codebook, img.shape, 5)
    # plt.figure()
    # plt.subplot(111)
    # plt.imshow(restored_img, cmap="gray")
    # plt.imsave("../output/pirate_decoded.png", restored_img, cmap="gray")
    # plt.show()

    # img = Image.open("../output/pirate_decoded.png").convert('L')
    # print(img.getbands()) # ('P',) 这种是有彩色的，而L是没有彩色的
    # img.save('../output/pirate_decoded.tif') # 转换后的进行保存
