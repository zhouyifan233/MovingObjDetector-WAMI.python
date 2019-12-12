"""
command example:
python WAMI_detector.py -I C:\WASABI-AngelFire-02\ -O C:\WASABI-output\ -N Models\ -M Customised
"""
import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
import TrainNetwork.BaseFunctions as basefunctions
import timeit
from SimpleTracker.KalmanFilter import KalmanFilter
from copy import copy
from MovingObjectDetector.BaseFunctions import TimePropagate, TimePropagate_, draw_error_ellipse2d
import hdf5storage


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-I ', '--InputFolder', type=str,
                        help='Input images folder')
    parser.add_argument('-O', '--OutputFolder', type=str,
                        help='Write images folder')
    parser.add_argument('-N', '--NNModelFolder', type=str, default="Models/",
                        help='The location of CNN')
    parser.add_argument('-M', '--Mode', type=str, default="WPAFB2009",
                        help='The running mode. Choose from WPAFB2009 or Customised')
    parser.add_argument('-T', '--BSThreshold', type=int, default=8,
                        help='Background Subtraction Threshold.')
    parser.add_argument('-NT', '--NumOfTemplate', type=int, default=5,
                        help='Number of templates for estimating background.')
    args = parser.parse_args()
    if args.Mode == "Customised":
        CustomisedDetection(args)


def CustomisedDetection(args):
    model_folder = args.NNModelFolder
    model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)
    model = (model_binary, aveImg_binary, model_regression, aveImg_regression)

    imagefolder = args.InputFolder
    print("Image folder: " + imagefolder)
    writeimagefolder = args.OutputFolder
    print("Output folder: " + writeimagefolder)
    num_of_template = args.NumOfTemplate
    print("Number of templates: " + str(num_of_template))

    # Read image folder
    filenames = os.listdir(imagefolder)
    filenames.sort()
    print(str(len(filenames)) + " images in the folder...")
    # Load background
    images = []
    for i in range(num_of_template):
        ReadImage = cv2.imread(imagefolder + filenames[i], cv2.IMREAD_GRAYSCALE)
        images.append(ReadImage)
    bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

    for i in range(num_of_template, len(filenames)):
        starttime = timeit.default_timer()
        # Read input image
        input_image = cv2.imread(imagefolder + filenames[i], cv2.IMREAD_GRAYSCALE)
        # Hs = bgt.doUpdateHomography(TransformationMatrices, frame_idx - 1)
        Hs = bgt.doCalculateHomography(input_image)

        bgt.doMotionCompensationAndValidArea(input_image, Hs, input_image.shape)
        CandiateCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels = bgt.doBackgroundSubtraction(
            input_image, thres=args.BSThreshold)
        print("background subtraction finished...")
        dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), CandiateCentres,
                                 BackgroundSubtractionProperties, BackgroundSubtractionLabels, model)
        Detections1, Detections2, RefinedCentres = dr.do_refine_bs()
        Detections1_for_img = [[ele["centre"][1], ele["centre"][0]] for ele in Detections1]
        Detections1_for_img = np.int64(np.asarray(Detections1_for_img))
        Detections2_for_img = [[ele["centre"][1], ele["centre"][0]] for ele in Detections2]
        Detections2_for_img = np.int64(np.asarray(Detections2_for_img))
        Detections3_for_img = [[ele[1], ele[0]] for ele in RefinedCentres]
        Detections3_for_img = np.int64(np.asarray(Detections3_for_img))
        print("Total number of detections: " + str(len(Detections1_for_img) + len(Detections2_for_img)))

        # plt.figure()
        output_image = copy(input_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
        for thisDetection in Detections1_for_img:
            if (thisDetection[0] > 0) and (thisDetection[0] < input_image.shape[1]) and (thisDetection[1] > 0)\
                    and (thisDetection[1] < input_image.shape[0]):
                cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 6, (0, 0, 200), 1)
        for thisDetection in Detections2_for_img:
            if (thisDetection[0] > 0) and (thisDetection[0] < input_image.shape[1]) and (thisDetection[1] > 0)\
                    and (thisDetection[1] < input_image.shape[0]):
                cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 6, (0, 0, 200), 1)
        #for thisDetection in Detections3_for_img:
        #    if (thisDetection[0] > 0) and (thisDetection[0] < input_image.shape[1]) and (thisDetection[1] > 0)\
        #            and (thisDetection[1] < input_image.shape[0]):
        #        cv2.circle(output_image, (thisDetection[0], thisDetection[1]), 1, (0, 215, 255), -1)
        print("Draw image finished...")
        print("Write image to " + writeimagefolder + filenames[i])
        cv2.imwrite(writeimagefolder + filenames[i], output_image)

        # update background
        bgt.updateTemplate(input_image)
        endtime = timeit.default_timer()
        print("Processing Time (Total): " + str(endtime - starttime) + " s... ")


if __name__ == "__main__":
    main()

