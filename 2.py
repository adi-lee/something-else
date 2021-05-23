import imageio
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import operator as opt
import os
import numpy as np

# fig = plt.figure()
# fig1=fig.add_subplot(1,2,1)
# p1=imageio.imread(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/0_1.jpg')
# plt.imshow(p1,cmap='Greys')
# plt.xlabel('p1')
# plt.show()
#
# # pic=np.random.randint(0,255,(28,28))
# # plt.imshow(pic,cmap='Greys')
# # plt.show()
#
# fig2=fig.add_subplot(1,2,2)
# p2=imageio.imread(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/0_1.jpg', as_gray=True)
# plt.imshow(p2,cmap='Greys')
# plt.xlabel('p2')
# plt.show()
# out = p2.resize((28,28),Image.ANTIALIAS)
# out.save(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/5.jpg')
# p3=imageio.imread(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/5.jpg')
# plt.imshow(p3)
# plt.xlabel('p3')
# plt.show()

# dp=np.random.randint(0,255,(28,28))
# plt.imshow(dp,cmap='Greys')
# plt.show()

import PIL
# im=PIL.Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/4.png')
# plt.imshow(im)
# plt.show()
# im.save(r'D:/PyCharm/pydata/some practice/number_identity/images/image_z/2.png',dpi=(28,28))
# im2 = PIL.Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/6.png')

# files = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/image_z')
# print(files)
# for file in files:
#     img = Image.open(os.path.join('my_folder', file))

# pic = PIL.Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/2.png')
# path = open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/0_1.txt', 'a')
# plt.imshow(pic)
# plt.show()
# width = pic.size[0]
# height = pic.size[1]
# print(width,height)
# print(pic.mode)
# for i in range(0, width):
#     for j in range(0, height):
#         C_RGB = pic.getpixel((i, j))
#         if C_RGB[0]+C_RGB[1]+C_RGB[2]> 0:
#             path.write('1')
#         elif C_RGB[0]+C_RGB[1]+C_RGB[2]== 0:
#             path.write('0')
#         else:
#             pass
#     path.write('\n')
# path.close()
# with open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/0_1.txt') as file1:
#     contents = file1.read()
#     print(contents)


from PIL import Image

img = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_z/3_3.png')
width, height = img.size
# print(width, height)
# 按比例缩小
out = img.resize((28, 28), Image.ANTIALIAS)
out.save('D:/PyCharm/pydata/some practice/number_identity/images/image_y/3_1.bmp', 'bmp')
img2 = Image.open('D:/PyCharm/pydata/some practice/number_identity/images/image_y/3_1.bmp')
plt.imshow(img2,cmap='Greys')
plt.show()


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
            if L> 0:
                arr.append(1)
            elif L == 0:
                arr.append(0)
            else:
                pass
        # arr.append('\n')
    # arr = map(eval, arr)
    # print(arr)
    return arr


def file_data1(file):
    arr = []
    pic = Image.open(file)
    width1 = pic.size[0]
    height1 = pic.size[1]
    # print(width,height)
    print(pic.mode)
    for i in range(0, width1):
        for j in range(0, height1):
            C_RGB = pic.getpixel((i, j))
            if C_RGB[0]+C_RGB[1]+C_RGB[2]> 0:
                arr.append(1)
            elif C_RGB[0]+C_RGB[1]+C_RGB[2] == 0:
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
    file_list = os.listdir('D:/PyCharm/pydata/some practice/number_identity/images/image_y')
    print(file_list)
    train_arr = np.zeros((len(file_list), 784))  # 28*28=784
    for i in range(0, len(file_list)):
        file1 = 'D:/PyCharm/pydata/some practice/number_identity/images/image_y/'+file_list[i]
        labels.append(file_list[i].split('_')[0])   # 确定手写数字体的真实数字
        # file_data(file)
        train_arr[i, :] = file_data1(file1)
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
    print(frame)
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

    a = img2txt()
    b = img2txt2()
    c = 0.0
    # print(b[1].shape[0])
    # print(b[0])
    for i in range(b[1].shape[0]):
        result = KNN(a[1], a[0], b[1][i], 4)
        print("该数字的真实值是： " + b[0][i])
        print("识别结果是： "+result)
        if result == b[0][i]:
            c += 1
    d = c / b[1].shape[0]

    print("识别成功率为： " + str(d))
