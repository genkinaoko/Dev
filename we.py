import cv2

video = cv2.VideoCapture(1)

while video.isOpened():

    ret, frame = video.read()
    if not ret: break
    
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    elif key == ord("s"):
        cv2.imwrite(r"/Users/genkitakasaki1/Desktop/Mycode/git/dev/resu/re.png", frame)

        

video.release()
cv2.destroyAllWindows()
