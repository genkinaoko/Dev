import cv2
import numpy as np
from matplotlib import pyplot as plt


def binarize(img):
    """画像を2値化する
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 2)
    plot_img(binary_img, 'binary_img')
    return binary_img


def noise_reduction(img):
    """ノイズ処理(中央値フィルタ)を行う
    """
    median = cv2.medianBlur(img, 9)
    plot_img(median, 'median')
    return median


def find_contours(img):
    """輪郭の一覧を得る
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def approximate_contours(img, contours):
    """輪郭を条件で絞り込んで矩形のみにする
    """
    height, width, _ = img.shape
    img_size = height * width
    approx_contours = []
    for i, cnt in enumerate(contours):
        arclen = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if arclen != 0 and img_size*0.02 < area < img_size*0.9:
            approx_contour = cv2.approxPolyDP(cnt, epsilon=0.05*arclen, closed=True)
            if len(approx_contour) == 4:
                approx_contours.append(approx_contour)
    return approx_contours


def draw_contours(img, contours, file_name):
    """輪郭を画像に書き込む
    """
    draw_contours_file = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255, 255), 10)
    plot_img(draw_contours_file, file_name)


def plot_img(img, file_name):
    """画像の書き出し
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(file_name)
    plt.show()
    #cv2.imwrite('./{}.png'.format(file_name), img)


def get_receipt_contours(img):
    """矩形検出までの一連の処理を行う
    """
    binary_img = binarize(img)
    binary_img = noise_reduction(binary_img)
    contours = find_contours(binary_img)
    approx_contours = approximate_contours(img, contours)
    draw_contours(img, contours, 'draw_all_contours')
    draw_contours(img, approx_contours, 'draw_rectangle_contours')


input_file = cv2.imread(r"C:\Users\_\Desktop\Code\Mycode\python\dev\pic\m9.JPG")
get_receipt_contours(input_file)