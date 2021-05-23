import numpy as np
import matplotlib.pylab as plt
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


def file_data(file):  # 将图片信息转换为0和1
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
                # L = 255 - L
                arr.append(L)
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
    print(arr)
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
    samples = random.sample(file_list, 3000)
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


# # sigmoid 函数的输出
# def sigmoid(x1):
#     return 1 / (1 + np.exp(-x1))
#
#
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
# helper to load data from PNG image files
import imageio


# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))  # 随机取即可,这个是一个经验函数
        # print(self.wih)
        # pow() 方法返回 xy（x 的 y 次方） 的值。
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # print(self.who.T)

        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

        # train the neural network

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T  # .T求逆矩阵，ndmin=2代表生成的矩阵是二维的
        print(inputs)
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # print(hidden_outputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # print(final_outputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

        # query the neural network

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        print(inputs)
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

training_data_file = open(r'G:\STUDY\2020-2021-2\大数据\mnist_test1.csv')

training_data_list = training_data_file.readlines()
print(training_data_list)
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 10  # 设置迭代次数

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # print("2")
        # split the record by the ',' commas
        all_values = record.split(',')
        # print(all_values)
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        # print(inputs)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        # print(targets)
        n.train(inputs, targets)
# load image data from png files into an array
print("loading ... my_own_images/2828_my_own_image.png")
# img_array = imageio.imread(r'D:\PyCharm\pydata\some practice\number_identity\images\image_y\3_1.bmp', as_gray=True)
#
# # reshape from 28x28 to list of 784 values, invert values
# # img_data = 255.0 - img_array.reshape(784)  # 之所以要进行这一步处理是因为要去除背景，使得测试数据与训练数据的像素矩阵一致。
# img_data = 255.0-img_array
# # then scale data to range from 0.01 to 1.0
# img_data = (img_data / 255.0 * 0.99) + 0.01
# print("min = ", numpy.min(img_data))
# print("max = ", numpy.max(img_data))
#
# # plot image
# matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')
#
# # query the network
# outputs = n.query(img_data)
# print(outputs)
#
# # the index of the highest value corresponds to the label
# label = numpy.argmax(outputs)
# print("network says ", label)

# 测试测试集的准确率
# f=open(r'C:\Users\Administrator.119V3UR3EO4VMWZ\Desktop\test10.txt')
# data=f.readlines()
# f.close()
#
# real=[]
# for i in data:
#    real.append(i[0])
# y=[]
# for i in data:
#    i=i.split(',')
#    outputs=n.query(numpy.asfarray(i[1:]))
#    y.append(numpy.argmax(outputs))

# 正确率为70%
message = input("请输入上传图片的名称： ")
user_upload_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_y'
img_load(message)  # 生成上传图片的规定尺寸大小的bmp格式
test_data = img2txt2(user_upload_route)
outputs = n.query(test_data[1])
print(outputs)
print(np.max(outputs))
label = np.argmax(outputs)
print(label)
print(test_data[0])

# test_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# ,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,
# 254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,
# 254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,
# 236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,
# 240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,
# 254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# outputs = n.query(test_data)
# print(outputs)
# label = np.argmax(outputs)
# print(label)


# message = input("请输入上传图片的名称： ")
# user_upload_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_y'
#
# im = Image.open(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/3_1.jpg')
# out = im.resize((28, 28), Image.ANTIALIAS)
# out.save(r'D:/PyCharm/pydata/some practice/number_identity/images/image_y/3_2.jpg')
# img_array = imageio.imread(r'D:\PyCharm\pydata\some practice\number_identity\images\image_y\3_2.jpg', as_gray=True)
# print(img_array)
# outputs = n.query(img_array)
# print(outputs)
# print(np.max(outputs))
# label = np.argmax(outputs)
# print(label)
# print(test_data[0])