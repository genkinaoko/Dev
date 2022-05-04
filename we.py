import cv2

video = cv2.VideoCapture(0)

while video.isOpened():

    ret, frame = video.read()
    if not ret: break
    
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break

    elif key == ord("s"):
        cv2.imwrite(r"C:\Users\_\Desktop\Code\Mycode\python\dev\pic\re.png", frame)

        

video.release()
cv2.destroyAllWindows()
