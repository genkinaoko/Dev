import cv2
import math
import numpy as np

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
            print('l=%d' % (l))

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

# この関数は、画像内のすべての正方形を描画します.
def drawSquares(image, squares ):
    retimg = cv2.polylines(image, squares, True, (0,255,0), thickness=2, lineType=cv2.LINE_8)
    cv2.imshow(wndname, image)
    return retimg

def main():
    names = [ '/Users/genkitakasaki1/Desktop/Mycode/git/dev/pic/ma.JPG',
        '/Users/genkitakasaki1/Desktop/Mycode/git/dev/pic/mb.JPG', '/Users/genkitakasaki1/Desktop/Mycode/git/dev/pic/mc.JPG', 
        '/Users/genkitakasaki1/Desktop/Mycode/git/dev/picm/md.JPG','/Users/genkitakasaki1/Desktop/Mycode/git/dev/pic/m8.JPG', 
        '/Users/genkitakasaki1/Desktop/Mycode/git/dev/pic/re.png', '/Users/genkitakasaki1/Desktop/Mycode/git/dev/pic/shape.png' ]

    squares = []
    for i, filename in enumerate(names):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None :
            print("Image file open error - ", filename)
            continue

        # 四角の角を見つける
        findSquares(image, squares)

        # 検出した四角を描画
        drawSquares(image, squares)
        c = cv2.waitKey()
        if c == 27:
            break
    return 0;

if __name__ == '__main__':
    main()