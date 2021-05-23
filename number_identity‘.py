import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import operator as opt
# pic = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/images_test1000/0_401.bmp')
# path = open('D:/PyCharm/pydata/some practice/number_identity/images/0_1.txt', 'a')
# plt.imshow(pic)
# plt.show()
# width = pic.size[0]
# height = pic.size[1]
# print(pic.mode)
# for i in range(0, width):
#     for j in range(0, height):
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

# 以上搞清楚了如何读取一个bmp位图，并将其转换为文本数据
# 上面的代码还将文本数据写入了txt文件中

# dirs = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/images_test1000')


# def file_data(filename):
#     arr = []
#     path = open(filename)
#     for i in range(0, 29):
#         line = path.readline()
#         for j in range(0, 29):
#             arr.append(line[j])
#     return arr
#
#
# def train_data():
#     lables = []
#     file_list = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/images_test1000')
#     trainarr = np.zeros((len(file_list), 1024))
#     print(trainarr)
#     for i in range(0, len(file_list)):
#         file = 'D:/PyCharm/pydata/some practice/number_identity/images/images_test1000'+file_list[i]
#         lables.append(file_list[i].split('_')[0])
#         trainarr[i, :] =file_data(file)
#     return trainarr, lables
#
#
# if __name__ == '__main__':
#     a = train_data()
#     print(a)
def file_data(file):
    arr = []
    pic = Image.open(file)
    width = pic.size[0]
    height = pic.size[1]
    # print(pic.mode)
    for i in range(0, width):
        for j in range(0, height):
            L = pic.getpixel((i, j))
            if L > 0:
                arr.append(1)
            elif L == 0:
                arr.append(0)
            else:
                pass
        # arr.append('\n')
    # arr = map(eval, arr)
    # print(arr)
    return arr


def img2txt():
    labels = []
    file_list = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/images4000')
    train_arr = np.zeros((len(file_list), 784))  # 28*28=784
    for i in range(0, len(file_list)):
        file1 = 'D:/PyCharm/pydata/some practice/number_identity/images/images4000/'+file_list[i]
        labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
        # file_data(file)
        train_arr[i, :] = file_data(file1)
    return labels, train_arr


def img2txt2():
    labels = []
    file_list = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/images_test1000')
    train_arr = np.zeros((len(file_list), 784))  # 28*28=784
    for i in range(0, len(file_list)):
        file1 = 'D:/PyCharm/pydata/some practice/number_identity/images/images_test1000/'+file_list[i]
        labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
        # file_data(file)
        train_arr[i, :] = file_data(file1)
    return labels, train_arr


def KNN(traindata, labels, testdata, k):
    dist = []
    dist1 = np.zeros(((testdata.shape[0]), 784))
    for j in range(traindata.shape[0]):
        dist_square = (traindata[j]-testdata) ** 2
        dist_square_sum = dist_square.sum()
        distances = dist_square_sum ** 0.5
        dist.append(distances)
    dist_dian = {'distance': dist, 'label': labels}
    frame = pd.DataFrame(dist_dian)
    frame = frame.sort_values(by="distance", ascending=True)
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

        # print(label)
        # print(frame)
        # dist1[i, :] = dist_dian
        # dist.sort()
        # dist_small = dist[:, k]
        # print(dist_small)

    # label_count = {}
    # for i in dist_small:


if __name__ == '__main__':

    a = img2txt()
    b = img2txt2()
    c = 0.0
    # print(b[1].shape[0])
    # print(b[0])
    for i in range(b[1].shape[0]):
        result = KNN(a[1], a[0], b[1][i], 3)
        print("该数字的真实值是： " + b[0][i])
        print("识别结果是： "+result)
        if result == b[0][i]:
            c += 1
    d = c / b[1].shape[0]

    print("识别成功率为： " + str(d))

    # KNN(a[1], a[0], b[1][0], 3)
    # print(a)


# def img2txt3():
#     labels = []
#     file_list = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/images_z')
#     train_arr = np.zeros((len(file_list), 784))  # 28*28=784
#     for i in range(0, len(file_list)):
#         file1 = 'D:/PyCharm/pydata/some practice/number_identity/images/images_z/'+file_list[i]
#         labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
#         # file_data(file)
#         train_arr[i, :] = file_data(file1)
#     return labels, train_arr


# def image_zhuan():
#
#     infile = 'D:/PyCharm/pydata/some practice/number_identity/images/images_z/0_1.bmp'
#     # outfile = 'D:/PyCharm/pydata/some practice/number_identity/images/images_z/0_2.bmp'
#     im = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/images_z/0_1.bmp')
#     # im = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/images_test1000/0_401.bmp')
#     (x, y) = im.size  # read image size
#     x_s = 28  # define standard width
#     y_s = 28  # calc height based on standard width
#     # out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
#     # out.save(outfile)


# if __name__ == '__main__':
#     image_zhuan()
#     a = img2txt()
#     b = img2txt2()
#     c = 1.0
#     # print(b[1].shape[0])
#     # print(b[0])
#     for i in range(b[1].shape[0]):
#         result = KNN(a[1], a[0], b[1][i], 3)
#         print("该数字的真实值是： " + b[0][i])
#         print("识别结果是： "+result)
#         if result == b[0][i]:
#             c += 1
#     d = c / b[1].shape[0]
#
#     print("识别成功率为： " + str(d))
