import numpy as np
import cv2
import matplotlib.pyplot as plt
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
import TrainNetwork.BaseFunctions as basefunctions

input_image_idx = 277

image_idx_offset = 8026480
num_of_template = 4
imagefolder = "E:/WASABI-AngelFire-02/"

image_idx_offset = 0
num_of_template = 4

model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels()

images = []
for i in range(num_of_template):
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % (input_image_idx+image_idx_offset+i-num_of_template), cv2.IMREAD_GRAYSCALE)
    images.append(ReadImage)
bgt = BackgroundModel(num_of_template=num_of_template, templates=images)
input_image = cv2.imread(imagefolder + "frame%06d.png" % (input_image_idx+image_idx_offset), cv2.IMREAD_GRAYSCALE)
Hs = bgt.doCalculateHomography(input_image)
bgt.doMotionCompensation(Hs, input_image.shape)
BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image, thres=10)

dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres, BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression, aveImg_regression)
refinedDetections, refinedProperties = dr.doMovingVehicleRefinement()
regressedDetections = dr.doMovingVehiclePositionRegression()
regressedDetections = np.asarray(regressedDetections)

plt.figure(11)
plt.imshow(np.repeat(np.expand_dims(input_image, -1), 3, axis=2))
plt.plot(BackgroundSubtractionCentres[:,0], BackgroundSubtractionCentres[:,1], 'g.')
plt.plot(refinedDetections[:,0], refinedDetections[:,1], 'y.')
plt.plot(np.int32(regressedDetections[:,0]), np.int32(regressedDetections[:,1]), 'r.')
plt.show()
