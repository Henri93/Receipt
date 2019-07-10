import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from transform import four_point_transform
from shapedetector import ShapeDetector

class ReceiptPreprocessor:
	""" 
    Helper Class responsible for aligning/cropping receipt image 
    """
	
	debug_mode = False

	def __init__(self, debug_mode=False):
		""" 
        Default Constructor

        Parameters: 
        debug_mode (bool): true for verbose logging

        """

		self.debug_mode = debug_mode
		pass


	def adaptive_thres(self, img):
		""" 
        Apply Adaptive Threshold

        Parameters: 
        img (image): image to apply adaptive threshold

        Returns:
        (image): image after applying threshold

        """

		img = cv2.medianBlur(img,5)
		return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

	def crop_rect(self, img, rect):
		""" 
        Crop rectangle from image

        Parameters: 
        img (image): image to crop from
        rect (Rectangle): the rectangle section to crop

        Returns:
        (image): image after applying crop

        """

		if debug_mode:
			print("rect: {}".format(rect))

		#convert rectangle into nice form
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		# get width and height of the detected rectangle
		width = int(rect[1][0])
		height = int(rect[1][1])

		src_pts = box.astype("float32")
		# corrdinate of the points in box points after the rectangle has been
		# straightened
		dst_pts = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")
		
		# the perspective transformation matrix
		M = cv2.getPerspectiveTransform(src_pts, dst_pts)

		# directly warp the rotated rectangle to get the straightened rectangle
		img = cv2.warpPerspective(img, M, (width, height))

		#rotate image back
		img = self.rotate_bound(img, rect[2])

		return img

	def find_receipt(self, img):
		""" 
        Find receipt in image using the following hueristic

        The receipt will be the largest white box

        Parameters: 
        img (image): image to find receipt in

        Returns:
        (image, Rectangle): Threshold image, Bounds of Receipt

        """

		#thresholding algorithm for binarization
		#make recipt white and background black
		blur = cv2.GaussianBlur(img,(51,51),0)
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		ret3,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		thresh = cv2.GaussianBlur(thresh, (1, 1), 0)
		
		#make img smaller so object detection is easier
		W = 500
		height, width = thresh.shape
		imgScale = W/width
		newX,newY = thresh.shape[1]*imgScale, thresh.shape[0]*imgScale

		#keep ratio for scaling detected contours up again
		resized = cv2.resize(thresh,(int(newX),int(newY)))
		ratio = thresh.shape[0] / float(resized.shape[0])

		#find the largest most white box in image
		largest = self.find_largest_white_rect(resized, img)
		rotrect = cv2.minAreaRect(largest)
		box = cv2.boxPoints(rotrect)
		box = np.int0(box)
	    
		#unwarp the boxed area
		# resized = four_point_transform(resized, box)

		#recalculate the white box for accuracy
		# largest = find_largest_white_rect(resized, img)

		#multiply the contour (x, y)-coordinates by the resize ratio,
		#then draw the contours and the name of the shape on the image
		largest = largest.astype("float")
		largest *= ratio
		largest = largest.astype("int")

		rotrect = cv2.minAreaRect(largest)
		box = cv2.boxPoints(rotrect)
		box = np.int0(box)

		if debug_mode:
			cv2.drawContours(thresh,[box],0,(255,0,0),10)

		return thresh, rotrect


	def find_largest_white_rect(self, thresh):
		""" 
        Find the largest white box in image

        Parameters: 
        thresh (image): image after threshold filtered

        Returns:
        (Rectangle): bounds of largest white box

        """

		#now detect white rectange in image
		contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

		sd = ShapeDetector()
		if debug_mode:
			print("Finding Contours...")
			#assume largest contor is outline?
			if len(contours) != 0:
				print("Found!")

		largest = contours[0]
		largest_mean = 0
		for cnt in contours:
			rotrect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rotrect)
			box = np.int0(box)

			cropImg = self.crop_rect(thresh, rotrect)

			crop_mean = cv2.mean(cropImg)[3]
			print(type(crop_mean))


			if cv2.contourArea(cnt) / (crop_mean+1) >= cv2.contourArea(largest) / (largest_mean+1):
				largest = cnt
				largest_mean = crop_mean
		return largest

	def apply_fft(self, img):
		""" 
        Apply FFT

        Parameters: 
        img (image): image to apply FFT

        Returns:
        (image): image after applying FFT

        """

		img_float32 = np.float32(img)
		dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
		dft_shift = np.fft.fftshift(dft)

		return 16*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

	def align_img_text(self, img):
		""" 
        Use FFT to create line align axis of rotation
        Then find degree of rotation of major lines and rotate image

        Parameters: 
        img (image): image to align text horizontally

        Returns:
        (double): angle to rotate the image by

        """
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rows,cols = img.shape

		fft = self.apply_fft(img)
		fft = cv2.GaussianBlur(fft,(5,5),0)

		threshold = 10
		edges = cv2.Canny(np.uint8(fft),23,23*3)
		lines = cv2.HoughLines(edges,1,np.pi/180,400)
		thetas = []
		if lines is None or len(lines) == 0:
			print("Relatively Straight")
			return 0

		# show original and fft image
		if self.debug_mode:
			plt.subplot(121),plt.imshow(img, cmap = 'gray')
			plt.title('Input Image'), plt.xticks([]), plt.yticks([])
			plt.subplot(122),plt.imshow(fft, cmap = 'gray')
			plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
			plt.show()

		for line in lines:
			for rho,theta in line:
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				deg = round(np.degrees(theta) % 45,2)
				if deg != 0 and deg != 90 and deg != 180:
					thetas.append(deg)
					# cv2.line(fft,(x1,y1),(x2,y2),(0,0,255),2)

		if len(thetas) == 0:
			print("No Theta Value Found")
			return 0

		calc_angles = []
		thetas = stats.mode(thetas)[0]

		#use 2 most frequent
		for i in range(min(len(thetas),2)):
			theta = thetas[i]
			calc_angles.append(theta)

		if self.debug_mode:
			print("Calc:", calc_angles)

		theta = np.mean(calc_angles)
		print("Angle of Rotation: " + str(theta))

		return theta

	def rotate_bound(self, image, angle):
		""" 
        Rotate an image by an angle

        Parameters: 
        img (image): image to rotate
        angle (double): angle of rotation

        Returns:
        (image): rotated image

        """

		# grab the dimensions of the image and then determine the
		# center
		(h, w) = image.shape[:2]
		(cX, cY) = (w // 2, h // 2)

		# grab the rotation matrix (applying the negative of the
		# angle to rotate clockwise), then grab the sine and cosine
		# (i.e., the rotation components of the matrix)
		M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])

		# compute the new bounding dimensions of the image
		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))

		# adjust the rotation matrix to take into account translation
		M[0, 2] += (nW / 2) - cX
		M[1, 2] += (nH / 2) - cY

		# perform the actual rotation and return the image
		return cv2.warpAffine(image, M, (nW, nH))

	def preprocess(self, img):
		""" 
        Preprocess a receipt image to extract text from

        Parameters: 
        img (image): image to process

        Returns:
        (image): processed image

        """
        
		#pre-rotate image to test
		fake_rotation = -23
		img = self.rotate_bound(img, fake_rotation)
		
		#get angle to rotate by
		theta = self.align_img_text(img)

		if theta != 0:
			img = self.rotate_bound(img, theta)
		else:
			#determined to be straight so remove pre-rotate
			img = self.rotate_bound(img, -fake_rotation)

		#find receipt
		# img_binary, receipt_outline = self.find_receipt(img)

		# box = cv2.boxPoints(receipt_outline)
		# box = np.int0(box)

		#crop to receipt
		# cv2.drawContours(img,[box],0,(0,255,0),10)
		# img = self.crop_rect(img, receipt_outline)
		# img = four_point_transform(img, box)

		return img