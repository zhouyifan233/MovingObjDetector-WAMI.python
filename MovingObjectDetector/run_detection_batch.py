import numpy as np
import cv2
import matplotlib.pyplot as plt
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
import TrainNetwork.BaseFunctions as basefunctions
import timeit
from SimpleTracker.KalmanFilter import KalmanFilter
from copy import copy
from MovingObjectDetector.BaseFunctions import TimePropagate, TimePropagate_, draw_error_ellipse2d
import hdf5storage

input_image_idx = 10

image_idx_offset = 0
num_of_template = 3
imagefolder = "E:/WPAFB-images/training/"
writeimagefolder = "E:/WPAFB-detections/savefig/"
model_folder = "C:/Users/yifan/Google Drive/PythonSync/wasabi-detection-python/Models/"
model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)
model = (model_binary, aveImg_binary, model_regression, aveImg_regression)

# load transformation matrices
matlabfile = hdf5storage.loadmat('C:/Users/yifan/Google Drive/PythonSync/wasabi-detection-python/Models/Data/TransformationMatrices_train.mat')
TransformationMatrices = matlabfile.get("TransMatrix")

# Load background
images = []
for i in range(num_of_template):
    frame_idx = input_image_idx+image_idx_offset+i-num_of_template
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    images.append(ReadImage)
bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

for i in range(20):
    starttime = timeit.default_timer()
    # Read input image
    frame_idx = input_image_idx+image_idx_offset+i
    input_image = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    Hs = bgt.doUpdateHomography(TransformationMatrices, frame_idx-1)
    # Hs = bgt.doCalculateHomography(input_image)

    bgt.doMotionCompensationAndValidArea(input_image, Hs, input_image.shape)
    CandiateCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels = bgt.doBackgroundSubtraction(input_image, thres=8, CompensateBrightness=False)
    print("background subtraction finished...")
    dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), CandiateCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels, model)
    Detections1, Detections2, RefinedCentres = dr.do_refine_bs()
    #refinedDetections = dr.doMovingVehicleRefinement()
    #print("CNN refinement finished...")
    #dr.associateCentresWithBackgroundSubtraction()
    #print("CNN refined regions of detections are associated with background subtraction region of detections...")
    #regressedDetections = dr.doMovingVehiclePositionRegression()
    #print("CNN regression subtraction finished...")
    Detections1_for_img = [[ele["centre"][1], ele["centre"][0]] for ele in Detections1]
    Detections1_for_img = np.int64(np.asarray(Detections1_for_img))
    Detections2_for_img = [[ele["centre"][1], ele["centre"][0]] for ele in Detections2]
    Detections2_for_img = np.int64(np.asarray(Detections2_for_img))
    #Detections3_for_img = [[ele[1], ele[0]] for ele in RefinedCentres]
    #Detections3_for_img = np.int64(np.asarray(Detections3_for_img))

    #plt.figure()
    output_image = copy(input_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    for thisDetection in Detections1_for_img:
        if thisDetection[0] > 0 and thisDetection[0] < input_image.shape[1] and thisDetection[1] > 0 and thisDetection[1] < input_image.shape[0]:
            cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 5, (0, 200, 0), 1)
    for thisDetection in Detections2_for_img:
        if thisDetection[0] > 0 and thisDetection[0] < input_image.shape[1] and thisDetection[1] > 0 and thisDetection[1] < input_image.shape[0]:
            cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 5, (0, 0, 200), 1)
    #for thisDetection in Detections3_for_img:
    #    if thisDetection[0] > 0 and thisDetection[0] < input_image.shape[1] and thisDetection[1] > 0 and thisDetection[1] < input_image.shape[0]:
    #        cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 1, (0, 150, 150), 1)
    print("Draw image finished...")
    print("Write image to " + writeimagefolder + "%05d.jpg"%frame_idx)
    cv2.imwrite(writeimagefolder + "%05d.jpg"%(frame_idx), output_image)
    """
    plt.figure()
    plt.imshow(np.repeat(np.expand_dims(input_image, -1), 3, axis=2))
    #plt.plot(BackgroundSubtractionCentres[:,0], BackgroundSubtractionCentres[:,1], 'g.')
    #plt.plot(refinedDetections[:,0], refinedDetections[:,1], 'y.')
    plt.plot(np.int32(regressedDetections[:,0]), np.int32(regressedDetections[:,1]), 'r.', markersize=3)
    plt.plot(np.int32(trackx), np.int32(tracky), 'yo', markersize=5)
    plt.show()
    """
    # update background
    bgt.updateTemplate(input_image)
    endtime = timeit.default_timer()
    print("Processing Time (Total): " + str(endtime - starttime) + " s... ")

