import cv2 
import numpy as np



def filter(img):
    # Conver BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

# open image.jpg
img = cv2.imread('image.jpg')
# filter image
res = filter(img)
# show image
cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
