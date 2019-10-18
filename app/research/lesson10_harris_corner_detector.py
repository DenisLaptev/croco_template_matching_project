import cv2
import numpy as np

#Works

# PATH_TO_IMAGE = '../resources/images_initial/img1.jpg'
# PATH_TO_IMAGE = '../resources/images_initial/img2_2016-03-01 21.42.11.jpg'
# PATH_TO_IMAGE = '../resources/images_initial/img3_20160630_160547.jpg'
PATH_TO_IMAGE = '../resources/images_initial/img4_20160630_160548.jpg'

img = cv2.imread(PATH_TO_IMAGE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(src=gray,
                       blockSize=2,
                       ksize=3,
                       k=0.04)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
