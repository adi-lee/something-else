import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
from bp_number_dientity import NeuralNetwork


import pandas as pd
import operator as opt
# pic = Image.open(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/0_1.bmp')
# # path = open(r'D:/PyCharm/pydata/some practice/number_identity/images/0_1.txt', 'a')
# # plt.imshow(pic)
# # plt.show()

# pic = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/images_test1000/4_490.bmp')
# path = open('D:/PyCharm/pydata/some practice/number_identity/images/0_1.txt', 'a')
# plt.imshow(pic)
# plt.show()
# width = pic.size[0]
# height = pic.size[1]
# print(pic.mode)
# for i in range(0, height):
#     for j in range(0, width):
#         L = pic.getpixel((i, j))
#         if L > 0:
#             path.write('1')
#         elif L == 0:
#             path.write('0')
#         else:
#             pass
#     path.write('\n')
# path.close()
# with open('D:/PyCharm/pydata/some practice/number_identity/images/0_1.txt') as file1:
#     contents = file1.read()
#     print(contents)


def file_data(file):
    arr = []
    pic = Image.open(file)
    width = pic.size[0]
    height = pic.size[1]
    # print(width,height)
    # print(pic.mode)
    for i in range(0, width):
        for j in range(0, height):
                L = pic.getpixel((i, j))
                arr.append(L)
    return arr


def img2txt(data_route):  #
    labels = []
    file_list = os.listdir(data_route)
    samples = random.sample(file_list, 4000)
    print(samples)
    train_arr = np.zeros((len(samples), 784))  # 28*28=784
    for i in range(0, len(samples)):
        file1 = data_route + '/' + samples[i]
        labels.append(samples[i].split('_')[0])   # 确定手写数字体的真实数字
        # file_data(file)
        train_arr[i, :] = file_data(file1)
    return labels, train_arr




def chu_shi_hua():
    train_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images4000'

    # 转换后的图像存放在image_y这里 用户上传到image_z
    train_data = img2txt(train_data_route)
    train_data_label = train_data[0]  # 训练数据集的真实数字标签
    train_data_txt = train_data[1]  # 训练数据集的文本数据

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    epochs = 10  # 迭代次数
    for e in range(epochs):
        # go through all records in the training data set

        for i in range(len(train_data_label)):
            inputs = (np.asfarray((train_data_txt[i])) /255 * 0.99) + 0.01
            # print(inputs)
            # print(inputs)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(train_data_label[i])] = 0.99
            # print(targets)
            n.train(inputs, targets)
    return n


def image_data_bian(images):
    arr = []
    for i in range(len(images[0])):
        for j in range(len(images[1])):
            arr.append(images[i][j])

    return arr


def find_number(images, network):
    images = image_data_bian(images)
    a = network.query(images)
    label = np.argmax(a)
    print("真实结果是：" + str(label))


frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(1)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 150)
net_work = chu_shi_hua()
while True:
    success, img = cap.read()
    cv2.imshow("result", img)
    # img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    # img = img.convert('L')
    find_number(img, net_work)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
