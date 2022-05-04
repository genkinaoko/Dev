import cv2
import matplotlib.pyplot as plt

from cv2 import THRESH_OTSU
from matplotlib.pyplot import gray 
img_file = r"C:\Users\_\Desktop\Code\Mycode\python\dev\pic\re.png"

origin_img = cv2.imread(img_file)
gray_img = cv2.imread(img_file, 0)
#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, img_th = cv2.threshold(img_th, 200, 255, cv2.THRESH_TOZERO_INV)
img_th = cv2.bitwise_not(img_th)
ret, img_th = cv2.threshold(img_th, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

draw_img = cv2.drawContours(origin_img.copy(), contours, -1, (0, 0, 255, 255), 10)
plt.imshow(cv2.cvtColor(draw_img, "a"))
plt.show()
