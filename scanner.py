from pyimagesearch import transform
import imutils
import cv2
import os
import argparse
import itertools
import numpy as np
import math
from scipy.spatial import distance as dist

class Scanner():
	def __init__(self, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
		'''
		MIN_QUAD_AREA_RATIO (float): A contour will be rejected if its corners 
                do not form a quadrilateral that covers at least MIN_QUAD_AREA_RATIO 
                of the original image. Defaults to 0.25.
        MAX_QUAD_ANGLE_RANGE (int):  A contour will also be rejected if the range 
                of its interior angles exceeds MAX_QUAD_ANGLE_RANGE. Defaults to 40.
		'''
		self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
		self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE

	def angle_between_vectors_degrees(self, u, v):
	    """Returns the angle between two vectors in degrees"""
	    return np.degrees(
	        math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

	def filter_corners(self, corners, min_dist=20):
	    """Filters corners that are within min_dist of others"""
	    def predicate(representatives, corner):
	        return all(dist.euclidean(representative, corner) >= min_dist
	                   for representative in representatives)

	    filtered_corners = []
	    for c in corners:
	        if predicate(filtered_corners, c):
	            filtered_corners.append(c)
	    return filtered_corners

	def get_angle(self, p1, p2, p3):
	    """
	    Returns the angle between the line segment from p2 to p1 
	    and the line segment from p2 to p3 in degrees
	    """
	    a = np.radians(np.array(p1))
	    b = np.radians(np.array(p2))
	    c = np.radians(np.array(p3))

	    avec = a - b
	    cvec = c - b

	    return self.angle_between_vectors_degrees(avec, cvec)

	def angle_range(self, quad):
	    """
	    Returns the range between max and min interior angles of quadrilateral.
	    The input quadrilateral must be a numpy array with vertices ordered clockwise
	    starting with the top left vertex.
	    """
	    tl, tr, br, bl = quad
	    ura = self.get_angle(tl[0], tr[0], br[0])
	    ula = self.get_angle(bl[0], tl[0], tr[0])
	    lra = self.get_angle(tr[0], br[0], bl[0])
	    lla = self.get_angle(br[0], bl[0], tl[0])

	    angles = [ura, ula, lra, lla]
	    return np.ptp(angles) 


	def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
		"""Returns True if the contour satisfies all requirements set at instantitation"""

		return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
		    and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

	def get_corners(self, img):
	    lsd = cv2.createLineSegmentDetector()
	    lines = lsd.detect(img)[0]

	    # massages the output from LSD
	    # LSD operates on edges. One "line" has 2 edges, and so we need to combine the edges back into lines
	    # 1. separate out the lines into horizontal and vertical lines.
	    # 2. Draw the horizontal lines back onto a canvas, but slightly thicker and longer.
	    # 3. Run connected-components on the new canvas
	    # 4. Get the bounding box for each component, and the bounding box is final line.
	    # 5. The ends of each line is a corner
	    # 6. Repeat for vertical lines
	    # 7. Draw all the final lines onto another canvas. Where the lines overlap are also corners

	    corners = []
	    if lines is not None:
	        # separate out the horizontal and vertical lines, and draw them back onto separate canvases
	        lines = lines.squeeze().astype(np.int32).tolist()
	        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
	        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
	        for line in lines:
	            x1, y1, x2, y2 = line
	            if abs(x2 - x1) > abs(y2 - y1):
	                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
	                cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
	            else:
	                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
	                cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

	        lines = []

	        # find the horizontal lines (connected-components -> bounding boxes -> final lines)
	        contours = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
	        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
	        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)

	        for contour in contours:
	            contour = contour.reshape((contour.shape[0], contour.shape[2]))
	            min_x = np.amin(contour[:, 0], axis=0) + 2
	            max_x = np.amax(contour[:, 0], axis=0) - 2
	            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
	            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
	            lines.append((min_x, left_y, max_x, right_y))
	            cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
	            corners.append((min_x, left_y))
	            corners.append((max_x, right_y))

	        # find the vertical lines (connected-components -> bounding boxes -> final lines)
	        contours = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
	        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
	        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)

	        for contour in contours:
	            contour = contour.reshape((contour.shape[0], contour.shape[2]))
	            min_y = np.amin(contour[:, 1], axis=0) + 2
	            max_y = np.amax(contour[:, 1], axis=0) - 2
	            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
	            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
	            lines.append((top_x, min_y, bottom_x, max_y))
	            cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
	            corners.append((top_x, min_y))
	            corners.append((bottom_x, max_y))

	        # find the corners
	        corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
	        corners += zip(corners_x, corners_y)

	    # remove corners in close proximity
	    corners = self.filter_corners(corners)
	    return corners


	def auto_canny(self, image, sigma=0.33):
		# compute the median of the single channel pixel intensities
		v = np.median(image)

		# apply canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)

		return edged

	def get_contours(self, image):
		MORPH = 9
		HOUGH = 25

		image_height, image_width, _ = image.shape

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7,7), 0)

		

		# dilate helps to remove potential holes between edge elements
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH,MORPH))
		dilated = cv2.dilate(gray, kernel)

		# find the edges
		#edged = cv2.Canny(dilated, 0, 84)
		edged = self.auto_canny(dilated)
		test_corners = self.get_corners(edged)

		approx_contours = []

		if len(test_corners) >= 4:
			quads = []

			for quad in itertools.combinations(test_corners, 4):
				points = np.array(quad)
				points = transform.order_points(points)
				points = np.array([[p] for p in points], dtype='int32')
				quads.append(points)

			# get top 5 quad
			quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
			# sort by their angle range
			quads = sorted(quads, key=self.angle_range)

			approx = quads[0]

			if self.is_valid_contour(approx, image_width, image_height):
			    approx_contours.append(approx)

 
		(_, cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

		#-------------------------CONTOUR DETECTION------------------------
		# loop over the contours
		for c in cnts:
		    # approximate the contour
		    approx = cv2.approxPolyDP(c, 80, True)
		    if self.is_valid_contour(approx, image_width, image_height):
		        approx_contours.append(approx)
		        break

		# If we did not find any valid contours, just use the whole image
		if not approx_contours:
		    top_right = (image_width, 0)
		    bottom_right = (image_width, image_height)
		    bottom_left = (0, image_height)
		    top_left = (0, 0)
		    screen_contour = np.array([[top_right], [bottom_right], [bottom_left], [top_left]])

		else:
		    screen_contour = max(approx_contours, key=cv2.contourArea)
		    
		return screen_contour.reshape(4, 2)
		#------------------------------------------------------------------



	def scan(self, image_path):
		NEW_HEIGHT = 500.0

		# load the image
		image = cv2.imread(image_path)
		ratio = image.shape[0] / NEW_HEIGHT
		original = image.copy()
		rescaled_image = imutils.resize(image, height=int(NEW_HEIGHT))

		screen_contour = self.get_contours(rescaled_image)

		# apply perspective transformation
		warped = transform.four_point_transform(original, screen_contour*ratio)

		# convert the warped image to grayscale
		gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

		#sharpen the image
		sharpen = cv2.GaussianBlur(gray, (0,0), 3)
		sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

		# apply adaptive threshold to get B/W effect
		thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

		# save the scanned image
		thresh = cv2.resize(thresh, (0,0), fx=0.3, fy=0.3)
		cv2.imshow('Scanned', thresh)
		cv2.waitKey(0)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image',
					help='path to the image')
	args = vars(ap.parse_args())

	scanner = Scanner()

	image_path = args['image']
	
	if image_path:
	    scanner.scan(image_path)
	else:
		print('Give Image Path!!!')

if __name__ == '__main__':
	main()
