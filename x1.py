from gettext import find
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

thresh = 50
N = 11
wndname = "Square Detection Demo"

# ベクトル間の角度の余弦(コサイン)を見つけます
# pt0-pt1およびpt0-pt2のなす角のコサインを取得
#
def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0,0] - pt0[0,0])
    dy1 = float(pt1[0,1] - pt0[0,1])
    dx2 = float(pt2[0,0] - pt0[0,0])
    dy2 = float(pt2[0,1] - pt0[0,1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) )
    return (dx1*dx2 + dy1*dy2)/ v

# 画像上で検出された一連の正方形を返します。
#
def findSquares(image, squares, areaThreshold=1000):
    squares.clear()
    gray0 = np.zeros(image.shape[:2], dtype=np.uint8)

    # down-scale and upscale the image to filter out the noise
    rows, cols, _channels = map(int, image.shape)
    pyr = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
    timg = cv2.pyrUp(pyr, dstsize=(cols, rows))

    # 画像のBGRの色平面で正方形を見つける
    for c in range(0, 3):
        cv2.mixChannels([timg], [gray0], (c, 0))

        # いくつかのしきい値レベルを試す
        for l in range(0, N):
            #print('l=%d' % (l))

            # l:ゼロしきい値レベルの代わりにCannyを使用します。
            # Cannyはグラデーションシェーディングで正方形を
            # キャッチするのに役立ちます
            if l == 0:
                # Cannyを適用
                # スライダーから上限しきい値を取得し、下限を0に設定します
                # （これによりエッジが強制的にマージ）
                #
                gray = cv2.Canny(gray0,thresh, 5)

                #Canny出力を拡張して、エッジセグメント間の潜在的な穴を削除します
                gray = cv2.dilate(gray, None)
            else:
                # apply threshold if l!=0:
                gray[gray0 >= (l+1)*255/N] = 0
                gray[gray0 < (l+1)*255/N] = 255

            # 輪郭をリストで取得
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                # 輪郭の周囲を取得
                arclen = cv2.arcLength(cnt, True)

                # 輪郭の近似
                approx = cv2.approxPolyDP(cnt, arclen*0.02, True)

                # 面積
                area = abs(cv2.contourArea(approx))

                #長方形の輪郭は、近似後に4つの角をもつ、
                #比較的広い領域
                #（ノイズの多い輪郭をフィルターで除去するため）
                #凸性(isContourConvex)になります。
                if approx.shape[0] == 4 and area > areaThreshold and cv2.isContourConvex(approx) :
                    maxCosine = 0

                    for j in range(2, 5):
                        # ジョイントエッジ間の角度の最大コサインを見つけます
                        cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
                        maxCosine = max(maxCosine, cosine)

                    # すべての角度の余弦定理が小さい場合（すべての角度が約90度）、
                    # 結果のシーケンスにquandrange頂点を書き込みます
                    if maxCosine < 0.3 :
                        squares.append(approx)

            #cs, h = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = []
    for i, cnt in enumerate(contours):

        arclen = cv2.arcLength(cnt, True)

        approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.005 * arclen, closed=True)

        approx_contours.append(approx_cnt)

    """
    left = max([squares[2:3].max() for contour in contours])
    right = min([squares[2:3,0,0].min() for contour in contours])
    upper = max([squares[2:3,0,1].max() for contour in contours])
    lower = min([squares[2:3,0,1].min() for contour in contours])
    """
    try:
        #min_x = list(squares[2:3])[0][0,0,0]
        min_x = min([list(squares[2:3])[0][0,0,0], list(squares[2:3])[0][1,0,0],list(squares[2:3])[0][2,0,0],list(squares[2:3])[0][3,0,0]])
        #min_y = list(squares[2:3])[0][0,0,1]
        min_y = min([list(squares[2:3])[0][0,0,1], list(squares[2:3])[0][1,0,1],list(squares[2:3])[0][2,0,1],list(squares[2:3])[0][3,0,1]])
        #max_x = list(squares[2:3])[0][2,0,0]
        max_x = max([list(squares[2:3])[0][0,0,0], list(squares[2:3])[0][1,0,0],list(squares[2:3])[0][2,0,0],list(squares[2:3])[0][3,0,0]])
        #max_y = list(squares[2:3])[0][2,0,1]
        max_y = max([list(squares[2:3])[0][0,0,1], list(squares[2:3])[0][1,0,1],list(squares[2:3])[0][2,0,1],list(squares[2:3])[0][3,0,1]])

    except IndexError:
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0
    """
    print(min_x)
    print(min_y)
    print(max_x)
    print(max_y)
    """
    #print("upper = {0}, lower = {1}, left = {2}, right = {3}".format(upper, lower, upper, lower))
    #return left, upper, right, lower;
    return min_y, max_y, min_x, max_x

def Convertion(img):
    p_original = np.float32()
    p_trans = np.float32()

    M = cv2.getPerspectiveTransform(p_original, p_trans)
    

# この関数は、画像内のすべての正方形を描画します.
def drawSquares(image, squares):
    retimg = cv2.polylines(image, squares[2:3], True, (0,255,0), thickness=2, lineType=cv2.LINE_8)
    cv2.imshow(wndname, image)
    #cv2.setMouseCallback('sample', onMouse)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return retimg

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def imgSave(img, min_y, max_y, min_x, max_x):
    try:
        img_edit = img[min_y:max_y, min_x:max_x]
        cv2.imwrite(r"/Users/genkitakasaki1/Desktop/Mycode/Git/Dev/pic/gazo.png", img_edit)
    except cv2.error:
        pass



video = cv2.VideoCapture(1)

squares =[]

#img = cv2.imread(r"/Users/genkitakasaki1/Desktop/Mycode/Git/Dev/pic/xxx.JPG")


while video.isOpened():
    ret, frame = video.read()
    if not ret: break
    min_y, max_y, min_x, max_x = findSquares(frame, squares)
    drawSquares(frame.copy(), squares)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if key == ord("s"): 
        imgSave(frame, min_y, max_y, min_x, max_x)
        #print(min_y, max_y, min_x, max_x)    

video.release()
#imgSave(img, min_y, max_y, min_x, max_x)
'''

path = r"/Users/genkitakasaki1/Desktop/Mycode/Git/Dev/pic/xxx.JPG"
img = cv2.imread(path)
last = img.copy()
squares = []
min_y, max_y, min_x, max_x = findSquares(img, squares)
#findSquares(img, squares)
drawSquares(img, squares)
imgSave(last, min_y, max_y, min_x, max_x)


img_edit = img.copy()[lower:upper, right:left]
cv2.imwrite(r"/Users/genkitakasaki1/Desktop/Mycode/Git/Dev/pic/resu.png", img_edit)
'''