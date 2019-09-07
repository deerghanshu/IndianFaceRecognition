from __future__ import print_function
import os
import sys
from collections import OrderedDict
import numpy as np
import cv2
import imutils
import dlib
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	overlay = image.copy()
	output = image.copy()
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		if name == "jaw":
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	return output
 
class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    def align(self, image, gray, rect):
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
 
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
 
        return output

#detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('PATH_TO_shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

def facealign(PATH):
	image = plt.imread(PATH)
	image = cv2.resize(image, (250,250))
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	faceAligned = fa.align(image, gray, dlib.rectangle(0,0,250,250))
	return faceAligned

#faceAligned = facealign('/content/drive/My Drive/Pytorch/HrithikRoshan/HirtikRoshan_203.jpg')
#plt.imshow(faceAligned)

#reconstruction
def createDataMatrix(images):
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	return data

def readImages(path):
	images = []
	for filePath in tqdm(sorted(os.listdir(path))):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in [".jpg", ".jpeg"]:
			imagePath = os.path.join(path, filePath)
			im = facealign(imagePath)

			if im is None :
				print("image:{} not read properly".format(imagePath))
			else :
				im = np.float32(im)/255.0
				images.append(im)
				imFlip = cv2.flip(im, 1);
				images.append(imFlip)
	numImages = int(len(images) / 2)
	if numImages == 0 :
		print("No images found")
		sys.exit(0)

	print(str(numImages) + " files read.")
	return images
 
def createNewFace(*args):
	output = averageFace
	for i in tqdm(range(0, NUM_EIGEN_FACES)):
		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eigenFaces[i] * weight)

	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

def resetSliderValues(*args):
	for i in range(0, NUM_EIGEN_FACES):
		cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
	createNewFace()

NUM_EIGEN_FACES = 20
MAX_SLIDER_VALUE = 255
dirName = "PATH_TO_IMG_FOLDER"
images = readImages(dirName)
sz = images[0].shape
data = createDataMatrix(images)

print("Calculating PCA ", end="...")
mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
print ("DONE")

averageFace = mean.reshape(sz)
eigenFaces = []

for eigenVector in eigenVectors:
  eigenFace = eigenVector.reshape(sz)
  eigenFaces.append(eigenFace)

cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
cv2.imshow("Result", output)
cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

sliderValues = []

for i in xrange(0, NUM_EIGEN_FACES):
sliderValues.append(MAX_SLIDER_VALUE/2)
cv2.createTrackbar( "Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2, MAX_SLIDER_VALUE, createNewFace)
cv2.setMouseCallback("Result", resetSliderValues);
cv2.waitKey(0)
cv2.destroyAllWindows()
