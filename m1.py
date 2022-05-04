import cv2

def main():
    #ライブラリのインポート
    
#画像読み込み
    img = cv2.imread(r"C:\Users\_\Desktop\Code\Mycode\python\dev\pic\re.png")

    #グレースケールへ変換
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #2値化 (画像、閾値、閾値を超えた場合に変更する値、2値化の方法)
    ret, img_th = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    img_th = cv2.bitwise_not(img_th)

    #輪郭検出
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #外接矩形を描写
    for cnt in contours:
        x, y, xx, yy = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+xx, y+yy), (0,0,255),10)

    cv2.imshow("bittest.jpg", img)    
    c = cv2.waitKey()
    if c == 27:
        return 0;

if __name__ == "__main__":
    main()