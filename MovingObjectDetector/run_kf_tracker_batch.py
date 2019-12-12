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

ROI_centre = [4000, 5000]
ROI_window = 2000
image_idx_offset = 0
num_of_template = 4
imagefolder = "C:/WPAFB-images/training/"
writeimagefolder = "C:/Workspace-python/savefig/"
model_folder = "C:/Users/yifan/Google Drive/PythonSync/wasabi-detection-python/Models/"
model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)

# load transformation matrices
matlabfile = hdf5storage.loadmat('C:/Users/yifan/Google Drive/PythonSync/wasabi-detection-python/Models/Data/TransformationMatrices_train.mat')
TransformationMatrices = matlabfile.get("TransMatrix")

# Load background
images = []
for i in range(num_of_template):
    frame_idx = input_image_idx+image_idx_offset+i-num_of_template
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    ReadImage = ReadImage[ROI_centre[1]-ROI_window:ROI_centre[1]+ROI_window+1, ROI_centre[0]-ROI_window:ROI_centre[0]+ROI_window+1]
    ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx-1][0])
    ROI_centre = [int(i) for i in ROI_centre]
    images.append(ReadImage)
bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

# initialise Kalman filter
kf = KalmanFilter(np.array([[2900], [2289], [0], [0]]), np.diag([900, 900, 400, 400]), 5, 6)
detections_all = []
for i in range(60):
    starttime = timeit.default_timer()
    # Read input image
    frame_idx = input_image_idx+image_idx_offset+i
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    input_image = ReadImage[ROI_centre[1]-ROI_window:ROI_centre[1]+ROI_window+1, ROI_centre[0]-ROI_window:ROI_centre[0]+ROI_window+1]
    ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx-1][0])
    ROI_centre = [int(i) for i in ROI_centre]

    Hs = bgt.doCalculateHomography(input_image)
    bgt.doMotionCompensation(Hs, input_image.shape)
    BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image, thres=10)

    dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres, BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression, aveImg_regression)
    refinedDetections, refinedProperties = dr.doMovingVehicleRefinement()
    regressedDetections = dr.doMovingVehiclePositionRegression()
    regressedDetections = np.asarray(regressedDetections)

    # Kalman filter update
    if i > 0:
        kf.TimePropagate(Hs[num_of_template-1])
        kf.predict()
        """"""
        # the id in the regressed detections
        regressionID = kf.NearestNeighbourAssociator(regressedDetections)
        # the id in the refinement detections (input to the CNN)
        refinementID = dr.refinedDetectionsID[regressionID]
        """"""
        kf.update()
        trackx = kf.mu_t[0,0]
        tracky = kf.mu_t[1,0]
        # propagate all detections
        detections_all = TimePropagate(detections_all, Hs[num_of_template - 1])
        detections_all.append(np.array([trackx, tracky]).reshape(2,1))
        print('Estimated State: ' + str(kf.mu_t.transpose()))
    else:
        trackx = kf.mu_t[0,0]
        tracky = kf.mu_t[1,0]
        detections_all.append(np.array([trackx, tracky]).reshape(2,1))

    # update background
    bgt.updateTemplate(input_image)

    #plt.figure()
    minx = np.int32(trackx-300)
    miny = np.int32(tracky-300)
    maxx = np.int32(trackx+301)
    maxy = np.int32(tracky+301)
    roi_image = np.repeat(np.expand_dims(input_image[miny:maxy, minx:maxx], -1), 3, axis=2)
    cv2.circle(roi_image, (301, 301), 10, (255, 0, 0), 1)
    validRegressedDetections = np.int32(copy(regressedDetections))
    validRegressedDetections[:, 0] = validRegressedDetections[:, 0] - minx
    validRegressedDetections[:, 1] = validRegressedDetections[:, 1] - miny
    for thisDetection in validRegressedDetections:
        if thisDetection[0] > 0 and thisDetection[0] < 600 and thisDetection[1] > 0 and thisDetection[1] < 600:
            cv2.circle(roi_image, (thisDetection[0], thisDetection[1]), 3, (100, 100, 0), -1)

    num_of_detections_all = len(detections_all)
    for idx in range(1, num_of_detections_all):
        point1x = np.int32(detections_all[idx-1][0,0]) - minx
        point1y = np.int32(detections_all[idx-1][1,0]) - miny
        point2x = np.int32(detections_all[idx][0,0]) - minx
        point2y = np.int32(detections_all[idx][1,0]) - miny
        cv2.line(roi_image, (point1x, point1y), (point2x, point2y), (0, 255, 0), 1)
    draw_error_ellipse2d(roi_image, (kf.mu_t[0]-minx, kf.mu_t[1]-miny), kf.sigma_t)
    #cv2.circle(input_image, (np.int32(trackx), np.int32(tracky)), 15, (255, 0, 0), 3)
    cv2.imwrite(writeimagefolder + "%05d.png"%i, roi_image)
    """
    plt.figure()
    plt.imshow(np.repeat(np.expand_dims(input_image, -1), 3, axis=2))
    #plt.plot(BackgroundSubtractionCentres[:,0], BackgroundSubtractionCentres[:,1], 'g.')
    #plt.plot(refinedDetections[:,0], refinedDetections[:,1], 'y.')
    plt.plot(np.int32(regressedDetections[:,0]), np.int32(regressedDetections[:,1]), 'r.', markersize=3)
    plt.plot(np.int32(trackx), np.int32(tracky), 'yo', markersize=5)
    plt.show()
    """
    endtime = timeit.default_timer()
    print("Processing Time (Total): " + str(endtime - starttime) + " s... ")


