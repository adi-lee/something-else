import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import operator as opt
import os
import numpy as np


# # cap = cv2.VideoCapture(1)
# # while 1:
# #     ret, frame = cap.read()
# #     cv2.imshow("capture", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# #
# # import numpy as np
# # import cv2
#
# # 人脸识别分类器
# faceCascade = cv2.CascadeClassifier(r'D:\PyCharm\pydata\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
#
# # 识别眼睛的分类器
# eyeCascade = cv2.CascadeClassifier(r'D:\PyCharm\pydata\venv\Lib\site-packages\cv2\data\haarcascade_eye.xml')
#
# # 开启摄像头
# cap = cv2.VideoCapture(1)
# ok = True
# result = []
# while ok:
#     # 读取摄像头中的图像，ok为是否读取成功的判断参数
#     ok, img = cap.read()
#     # 转换成灰度图像
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 人脸检测
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(32, 32)
#     )
#
#     # 在检测人脸的基础上检测眼睛
#     for (x, y, w, h) in faces:
#         fac_gray = gray[y: (y+h), x: (x+w)]
#
#         eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)
#
#         # 眼睛坐标的换算，将相对位置换成绝对位置
#         for (ex, ey, ew, eh) in eyes:
#             result.append((x+ex, y+ey, ew, eh))
#
#     # 画矩形
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#     # for (ex, ey, ew, eh) in result:
#     #     cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
#
#     cv2.imshow('video', img)
#
#     k = cv2.waitKey(1)
#     if k == 27:    # press 'ESC' to quit
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# def cam_load():
#     cap = cv2.VideoCapture(1)  # 计算机自带的摄像头为0，外部设备为1
#     i = 0
#     while 1:
#         ret, frame = cap.read()  # ret:True/False,代表有没有读到图片  frame:当前截取一帧的图片
#         cv2.imshow("capture", frame)
#
#         # if (cv2.waitKey(1) & 0xFF) == ord('s'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
#         # if (cv2.waitKey(1) & 0xFF) == ord('s'):
#         #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
#         #     gray = cv2.resize(gray, (320, 240))  # 图像大小为320*240
#         #     # cv2.imwrite('F:/dlib-19.16/dlib-19.16/tools/imglab/build/images/%d.jpg' % i, gray)
#         #     cv2.imwrite('D:/PyCharm/pydata/some practice/number_identity/images/image_x/%d.png' % i, gray)
#         #     i += 1
#         cv2.waitKey(100)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
#         gray = cv2.resize(gray, (28, 28))  # 图像大小为320*240
#         # cv2.imwrite('F:/dlib-19.16/dlib-19.16/tools/imglab/build/images/%d.jpg' % i, gray)
#         cv2.imwrite('D:/PyCharm/pydata/some practice/number_identity/images/image_x/%d.png' % i, gray)
#         i += 1
#         if i == 10:
#             cam_load_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_x'
#             test_data = img2txt2(cam_load_route)
#             train(test_data)
#             i = 0
#         if (cv2.waitKey(1) & 0xFF) == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


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


def file_data1(images):
    arr = []
    width = images[0]
    height = images[1]
    for i in range(len(images[0])):
        for j in range(len(images[1])):
                L = images[i][j]
                if L > 0:
                    arr.append(1)
                elif L == 0:
                    arr.append(0)
                else:
                    pass
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


def img2txt2(data_route):  #
    labels = []
    file_list = os.listdir(data_route)
    train_arr = np.zeros((len(file_list), 784))  # 28*28=784
    for i in range(0, len(file_list)):
        file1 = data_route + '/' + file_list[i]
        pic = Image.open(file1)
        plt.imshow(pic)
        plt.show()
        train_arr[i, :] = file_data(file1)
    return labels, train_arr


def KNN(train_data, labels, test_data, k):
    dist = []
    # dist1 = np.zeros(((test_data.shape[0]), 784))
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


def cam_load():
    frame_width = 640
    frame_height = 480
    cap = cv2.VideoCapture(1)
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    cap.set(10, 150)

    while True:
        success, img = cap.read()
        cv2.imshow("result", img)
        # img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        # img = img.convert('L')
        img_data = file_data1(img)
        train1(img_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def chu_shi_hua():
    global test_data
    global model_choose
    train_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images4000'
    test_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images_test1000'
    user_upload_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_y'
    cam_load_route = 'D:/PyCharm/pydata/some practice/number_identity/images/image_x'
    # 转换后的图像存放在image_y这里 用户上传到image_z
    print("--------------------------------------")
    print("--------基于KNN算法的手写数字识别-------")
    print("模式选择：")
    print("测试上传图片请输入1，测试训练模型请输入2，调用摄像头进行实时识别请输入3")
    model_choose = input()
    if model_choose == "1" or model_choose == "2" or model_choose == "3":
        if model_choose == "1":
            message = input("请输入上传图片的名称： ")
            img_load(message)  # 生成上传图片的规定尺寸大小的bmp格式
            test_data = img2txt(user_upload_route)
        elif model_choose == "2":
            test_data = img2txt(test_data_route)
        elif model_choose == "3":
            cam_load()

        else:
            print("模式选择错误，请重选")
    return test_data


def train(test_data1):
    c = 0.0
    train_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images4000'
    train_data = img2txt(train_data_route)
    for i in range(test_data1[1].shape[0]):
        result = KNN(train_data[1], train_data[0], test_data1[1][i], 4)
        if model_choose != "3":
            print("该数字的真实值是： " + test_data1[0][i])
            if result == test_data1[0][i]:
                c += 1
        d = c / test_data1[1].shape[0]
        print("识别结果是： " + result)

    print("识别成功率为： " + str(d))


def train1(arr):
    train_data_route = 'D:/PyCharm/pydata/some practice/number_identity/images/images_test1000'
    train_data = img2txt(train_data_route)
    result = KNN(train_data[1], train_data[0], arr, 4)
    print("识别结果是： " + result)


if __name__ == '__main__':
    test_data1 = chu_shi_hua()
    train(test_data1)















