import numpy as np
import argparse
import cv2,cv2.cv as cv

def ratio(x,y):
    if((x/y)>0.4):
        return "25 seconds"
    else:
        return "15 seconds"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "/home/naveen/Downloads/")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)


#RGB values
boundaries = [([2,90,2],[250,250,250])]

for (lower, upper) in boundaries:
# create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    mask = cv2.inRange(image, lower, upper)
    mask1 = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(image, image, mask = mask)
    output1 = cv2.bitwise_and(hsv, hsv, mask = mask1)

#Find the size of image and individual pixel data
    height,width,channels=image.shape
    print height,width
    pixdata=[]
    for i in range(0,height):
        for j in range(0,width):
            imag=output1[i,j]
            pixdata.append((imag[0],imag[1],imag[2]))

#Find the number of valid and invalid pixels
    valid=0
    invalid=0
    for each in pixdata:
        if ((each[0]<2) or (each[1]<90) or (each[2]<2)) or((each[0]>250) or (each[1]>250) or (each[2]>250)):
            valid=valid+1
        else:
            invalid=invalid+1
    print invalid,valid

    p=ratio(valid,invalid)
    print p

    cv2.imshow("images", np.hstack([image, output]))
    cv2.imshow("hsv", np.hstack([hsv, output1]))
    #cv2.imshow("mask",mask)
    #cv2.imshow("mask1",mask1)
    cv2.waitKey(0)
