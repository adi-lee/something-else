import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import operator as opt
import os
import numpy as np


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
                if L > 0:
                    arr.append(1)
                elif L == 0:
                    arr.append(0)
                else:
                    pass
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


def img2txt(data_route):  #
    labels = []
    file_list = os.listdir(data_route)
    train_arr = np.zeros((len(file_list), 784))  # 28*28=784
    for i in range(0, len(file_list)):
        file1 = data_route + '/' + file_list[i]
        labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
        # file_data(file)
        train_arr[i, :] = file_data(file1)
    return labels, train_arr


def KNN(train_data, labels, test_data, k):
    dist = []
    dist1 = np.zeros(((test_data.shape[0]), 784))
    for j in range(train_data.shape[0]):
        dist_square = (train_data[j]-test_data) ** 2
        dist_square_sum = dist_square.sum()
        distances = dist_square_sum ** 0.5
        dist.append(distances)
    dist_dian = {'distance': dist, 'label': labels}
    frame = pd.DataFrame(dist_dian)
    frame = frame.sort_values(by="distance", ascending=True)
    # print(frame)
    frame1 = frame.head(k)
    label = frame1['label']
    label = np.array(label)
    label = label.tolist()
    # print(label)
    labels_count = {}
    for m in range(k):
        label_1 = label[m]
        labels_count[label_1] = labels_count.get(label_1, 0) + 1  # 次数加一，使用字典的get方法，第一次出现时默认值是0
    sorted_count = sorted(labels_count.items(), key=opt.itemgetter(1), reverse=True)
    # print(sorted_count[0][0])
    return sorted_count[0][0]


if __name__ == '__main__':
    c = 0.0
    train_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images4000'
    test_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images_test1000'
    user_upload_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_y'
    # 转换后的图像存放在image_y这里 用户上传到image_z
    print("--------------------------------------")
    print("--------基于KNN算法的手写数字识别-------")
    print("模式选择：")
    print("测试上传图片请输入1，测试训练模型请输入2")
    model_choose = input()
    if model_choose == "1" or model_choose == "2":
        if model_choose == "1":
            message = input("请输入上传图片的名称： ")
            img_load(message)  # 生成上传图片的规定尺寸大小的bmp格式
            test_data = img2txt(user_upload_route)
        elif model_choose == "2":
            test_data = img2txt(test_data_route)
        train_data = img2txt(train_data_route)
        for i in range(test_data[1].shape[0]):
            result = KNN(train_data[1], train_data[0], test_data[1][i], 10)
            print("该数字的真实值是： " + test_data[0][i])
            print("识别结果是： " + result)
            if result == test_data[0][i]:
                c += 1
        d = c / test_data[1].shape[0]

        print("识别成功率为： " + str(d))
    else:
        print("模式选择错误，请重选")











