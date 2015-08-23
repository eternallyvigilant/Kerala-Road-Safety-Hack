import numpy as np
import cv2

cap = cv2.VideoCapture('thermal.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray=frame
    height,width,channels=gray.shape


#c1,r1=0


# define range of red color in HSV
    lower_blue = np.array([2,90,2],dtype = "uint8")
    upper_blue = np.array([255,255,255],dtype = "uint8")

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(gray, lower_blue, upper_blue)
    output = cv2.bitwise_and(gray, gray, mask = mask)

#Find the size of image and individual pixel data



    cv2.imshow('frame',output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
