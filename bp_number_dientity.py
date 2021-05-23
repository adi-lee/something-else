import scipy.special
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import pandas as pd
import operator as opt
import os
import numpy as np
import random


def img_load(img_name):  # 加载并处理用户上传的图片，并将其转换为28*28的bmp文件
    img = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/' + img_name)
    string1 = img_name
    print(string1)
    arr1 = []
    for i in range(len(string1)):
        if string1[i] != '.':
            arr1.append(string1[i])
        else:
            break
    string2 = ''.join(arr1)
    # print(string2)  #  提取图片字母的名称，以便后续命名
    # width, height = img.size
    # # print(width, height)
    # # 按固定尺寸缩小
    out = img.resize((28, 28), Image.ANTIALIAS)
    out = out.convert('L')
    out.save('D:/PyCharm/pydata/some practice/number_identity/images/image_y/' + string2 + '.bmp', 'bmp')
    img2 = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_y/' + string2 + '.bmp')
    plt.imshow(img2, cmap='Greys')
    plt.show()  # 转换图片格式并重命名


def file_data(file):
    arr = []
    pic = Image.open(file)
    width = pic.size[0]
    height = pic.size[1]
    # print(width,height)
    # print(pic.mode)
    for i in range(0, width):
        for j in range(0, height):
            if pic.mode == 'L':
                L = pic.getpixel((i, j))
                arr.append(L)
                # if L > 0:
                #     arr.append(1)
                # elif L == 0:
                #     arr.append(0)
                # else:
                #     pass
            elif pic.mode == 'RGB':
                C_RGB = pic.getpixel((i, j))
                if C_RGB[0] + C_RGB[1] + C_RGB[2] > 0:
                    arr.append(1)
                elif C_RGB[0] + C_RGB[1] + C_RGB[2] == 0:
                    arr.append(0)
                else:
                    pass
        # arr.append('\n')
    # arr = map(eval, arr)
    # print(arr)
    return arr


# def img2txt(data_route):  #
#     labels = []
#     file_list = os.listdir(data_route)
#     train_arr = np.zeros((len(file_list), 784))  # 28*28=784
#     for i in range(0, len(file_list)):
#         file1 = data_route + '/' + file_list[i]
#         labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
#         # file_data(file)
#         train_arr[i, :] = file_data(file1)
#     return labels, train_arr


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


def img2txt2(data_route):
    labels = []
    file_list = os.listdir(data_route)
    train_arr = np.zeros((len(file_list), 784))  # 28*28=784
    for i in range(0, len(file_list)):
        file1 = data_route + '/' + file_list[i]
        labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
        # file_data(file)
        train_arr[i, :] = file_data(file1)
    return labels, train_arr


class NeuralNetwork:

    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # normal(loc, scale, size)   产生具有正态分布的数组, loc均值, scale标准差, size形状
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        # print(self.wih[0][0])
        # pow() 方法返回 xy（x 的 y 次方） 的值。
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # print(self.who.T)

        # learning rate
        self.lr = learning_rate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

        # train the neural network

    def train(self, inputs_list, targets_list, arr):

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T  # .T求逆矩阵，ndmin=2代表生成的矩阵是二维的
        # print(inputs)
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # print(hidden_outputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # print(final_outputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        arr.append(output_errors[0][0])
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))  # np.transpose()求转置

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))
        # arr.append(self.wih[0][0])

        pass

        # query the neural network

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def chu_shi_hua():
    train_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images4000'  # 训练集路径

    train_data = img2txt(train_data_route)  # 将训练集转换为文本
    train_data_label = train_data[0]  # 训练数据集的真实数字标签
    train_data_txt = train_data[1]  # 训练数据集的文本数据

    input_nodes = 784  # 输入层节点个数
    hidden_nodes = 200  # 隐藏层节点个数
    output_nodes = 10  # 输出层节点个数
    learning_rate = 0.1  # 学习率
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)  # 网络训练
    epochs = 20  # 迭代次数
    arr = []
    for e in range(epochs):
        # go through all records in the training data set

        for i in range(len(train_data_label)):
            inputs = (np.asfarray((train_data_txt[i])) /255 * 0.99) + 0.01  # 将训练集数据归一化
            # print(inputs)
            # print(inputs)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(train_data_label[i])] = 0.99  # 将真实数字标签与与图像输出分类对应
            # print(targets)
            n.train(inputs, targets, arr)  # 训练模型
    print(arr)
    plt.plot(arr)
    plt.ylabel("error")
    plt.xlabel("Training times")
    plt.show()

    return n  # 将BP网络类对象返回

# test_data = img2txt2(test_data_route)
# test_data_label = test_data[0]
# test_data_txt = test_data[1]
# output = []
# c = 0.0
# for i in range(len(test_data_label)):
#     a = n.query(test_data_txt[i])
#     label = np.argmax(a)
#     output.append(label)
#     print("识别结果是：" + str(label))
#     print("真实结果是：" + test_data_label[i])
#     if str(label) == test_data_label[i]:
#         c += 1
# d = c/len(test_data_label)
# print("识别准确率：" + str(d))


# while 1:
#
#     message = input("请输入上传图片的名称： ")
#     img_load(message)  # 生成上传图片的规定尺寸大小的bmp格式
#     test_data = img2txt2(user_upload_route)
#     print(test_data[1])
#     outputs = n.query(test_data[1])
#     print(outputs)
#     print(np.max(outputs))
#     label = np.argmax(outputs)
#     print(label)
#     print(test_data[0])


if __name__ == '__main__':
    n = chu_shi_hua()  # 初始化得到训练好的BP神经网络
    test_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images_test1000'
    user_upload_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_y'
    print("--------------------------------------")
    print("--------基于BP神经网络的手写数字识别-------")
    print("模式选择：")
    print("测试上传图片请输入1，测试训练模型请输入2")
    model_choose = input()  # 选择手写数字识别的模式
    if model_choose == "1" or model_choose == "2":
        if model_choose == "1":
            message = input("请输入上传图片的名称： ")
            img_load(message)  # 生成上传图片的规定尺寸大小的bmp格式
            test_data = img2txt2(user_upload_route)
        elif model_choose == "2":
            test_data = img2txt2(test_data_route)
        test_data_label = test_data[0]
        test_data_txt = test_data[1]
        output = []
        c = 0.0
        for i in range(len(test_data_label)):
            a = n.query(test_data_txt[i])
            label = np.argmax(a)
            output.append(label)
            print("识别结果是：" + str(label))
            print("真实结果是：" + test_data_label[i])
            if str(label) == test_data_label[i]:
                c += 1  # 识别正确则加一
        d = c / len(test_data_label)  # 识别正确数/识别总数得到识别准确率
        print("识别准确率：" + str(d))
    else:
        print("模式选择错误，请重选")
