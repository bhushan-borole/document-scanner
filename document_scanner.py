from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image
# compute the ratio of the old height to new height
image = cv2.imread(args['image'])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

#--------------- STEP 1: Edge Detection ---------------------

# grayscale, blur and find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

#------------------------------------------------------------

#--------------- STEP 2: Contour Detection ------------------

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
screenCnt = None

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02*peri, True)

	if len(approx) == 4:
		screenCnt = approx
		break

#-------------------------------------------------------------


#--------------- STEP 3: Apply Perspective Transform----------

# apply the four point transform to obtain a top-down view of original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2)*ratio)

# convert the warped to grayscale, then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method='gaussian')
warped = (warped > T).astype('uint8') * 255

#-------------------------------------------------------------

cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)




