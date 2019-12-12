import numpy as np
import TrainNetwork.BaseFunctions as bf
from sklearn.neighbors import NearestNeighbors
import skimage.measure as measure
import cv2
import matplotlib.pyplot as plt


class DetectionRefinement:

    def __init__(self, input_image, compensatedImages, CandidateCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels, model):
        self.num_of_template = len(compensatedImages)
        self.img_t = input_image
        self.img_tminus1 = compensatedImages[self.num_of_template-1]
        self.img_tminus2 = compensatedImages[self.num_of_template-2]
        self.img_tminus3 = compensatedImages[self.num_of_template-3]
        self.CandidateCentres = CandidateCentres
        self.bgProperties = BackgroundSubtractionProperties
        self.bgLabels = BackgroundSubtractionLabels
        self.model_binary = model[0]
        self.aveImg_binary = model[1]
        self.model_regression = model[2]
        self.aveImg_regression = model[3]
        self.RefinedCentres = []
        self.AcceptedDetectionProp = []
        self.PendingForRegressionProp = []
        self.PredictedPositions = []

    def refine_centres(self):
        img_shape = self.img_t.shape
        width = img_shape[1]
        height = img_shape[0]
        num_points = len(self.CandidateCentres)
        X = np.ndarray((num_points, 4, 21, 21), dtype=np.float32)
        mask1 = np.zeros(num_points, dtype=np.bool)
        for i, thisdetection in enumerate(self.CandidateCentres):
            minr = np.int32(np.round(thisdetection[0] - 10))
            minc = np.int32(np.round(thisdetection[1] - 10))
            maxr = np.int32(np.round(thisdetection[0] + 11))
            maxc = np.int32(np.round(thisdetection[1] + 11))
            if minr > 0 and minc > 0 and maxr < height and maxc < width:
                data_t = np.reshape(self.img_t[minr:maxr, minc:maxc], (1, 21, 21))
                data_tminus1 = np.reshape(self.img_tminus1[minr:maxr, minc:maxc], (1, 21, 21))
                data_tminus2 = np.reshape(self.img_tminus2[minr:maxr, minc:maxc], (1, 21, 21))
                data_tminus3 = np.reshape(self.img_tminus3[minr:maxr, minc:maxc], (1, 21, 21))
                X[i] = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
                mask1[i] = True
        # mask 1 relates to the detections that available to CNN
        X = X[mask1, ...]
        ValidCandidateCentres = self.CandidateCentres[mask1, ...]
        X, _ = bf.DataNormalisationZeroCentred(X, self.aveImg_binary)
        predictResults = self.model_binary.predict(X, batch_size=5000, verbose=0)
        # mask2 relates to the filtered detections (by mask1) that are accepted by CNN
        mask2 = np.zeros(len(predictResults), dtype=np.bool)
        for idx in range(len(predictResults)):
            thisResult = predictResults[idx]
            if thisResult[0] > 0.5:
                mask2[idx] = True
        RefinedCentres = ValidCandidateCentres[mask2, ...]
        # mask 3 mapps raw background subtraction detections to CNN accepted detections
        mask3 = mask1
        mask3[mask1] = mask2
        self.RefinedCentres = RefinedCentres
        return RefinedCentres

    def associate_centres_bs(self):
        AcceptedDetectionProp = []
        PendingForRegressionProp = []
        CandidateRegion = np.zeros(self.img_t.shape, dtype=np.uint8)
        for detection_ele in self.RefinedCentres:
            tmp_r = detection_ele[0]
            tmp_c = detection_ele[1]
            CandidateRegion[tmp_r-3:tmp_r+4, tmp_c-3:tmp_c+4] = 1
        CandidateRegion_labels = measure.label(CandidateRegion, connectivity=1)
        CandidateRegion_Properties = measure.regionprops(CandidateRegion_labels)
        for prop_ele in CandidateRegion_Properties:
            RegionSize = prop_ele.area
            RegionCoords = prop_ele.coords
            Region_on_Bg = self.bgLabels[RegionCoords[:, 0], RegionCoords[:, 1]]
            Region_on_Bg = Region_on_Bg[Region_on_Bg != 0]
            if Region_on_Bg.size > 0:
                UniqueLabels = np.unique(Region_on_Bg)
                if (UniqueLabels.size == 1) and (RegionSize <= 200):
                    UniqueLabels = UniqueLabels[0]-1
                    if RegionSize / self.bgProperties[UniqueLabels].area >= 0.5:
                        AcceptedDetectionProp.append(self.bgProperties[UniqueLabels])
                    else:
                        AcceptedDetectionProp.append(prop_ele)
                else:
                    PendingForRegressionProp.append(prop_ele)
            else:
                AcceptedDetectionProp.append(prop_ele)
        print("Num of accepted Detections: " + str(len(AcceptedDetectionProp)))
        print("Num of pending Detections (for regression): " + str(len(PendingForRegressionProp)))
        self.AcceptedDetectionProp = AcceptedDetectionProp
        self.PendingForRegressionProp = PendingForRegressionProp
        return AcceptedDetectionProp

    def predict_moving_obj_locations(self):
        PredictedPositions_final = []
        win_size = 22
        net_dim = 2 * win_size + 1
        T = 0.20
        img_shape = self.img_t.shape
        for PendingForRegressionProp_ele in self.PendingForRegressionProp:
            PredictedPositions_ele_raw = []
            BoundingBox = PendingForRegressionProp_ele.bbox
            centre = PendingForRegressionProp_ele.centroid
            if (BoundingBox[2]-BoundingBox[0] <= net_dim/2) and (BoundingBox[3]-BoundingBox[1] <= net_dim/2):
                min_r = np.int32(centre[0] - win_size)
                max_r = np.int32(centre[0] + win_size)
                min_c = np.int32(centre[1] - win_size)
                max_c = np.int32(centre[1] + win_size)
                if (min_r > 0) and (min_c > 0) and (max_r < img_shape[0]) and (max_c < img_shape[1]):
                    data_t = np.reshape(self.img_t[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                    data_tminus1 = np.reshape(self.img_tminus1[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                    data_tminus2 = np.reshape(self.img_tminus2[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                    data_tminus3 = np.reshape(self.img_tminus3[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                    X = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
                    X = np.expand_dims(X, axis=0)
                    X, _ = bf.DataNormalisationZeroCentred(X, self.aveImg_regression)
                    RegressionResult = self.model_regression.predict(X, batch_size=1, verbose=0)
                    RegressionResult = cv2.resize(np.reshape(RegressionResult, (15, 15)), (45, 45))
                    PredictedPositions_ele_raw.extend(extractPointsFromRegressionImage(RegressionResult, min_r, min_c, T))
            else:
                start_r = min([BoundingBox[0] + win_size/2, centre[0]])
                end_r = max([BoundingBox[2], centre[0]])
                start_c = min([BoundingBox[1] + win_size / 2, centre[1]])
                end_c = max([BoundingBox[3], centre[1]])
                current_r = start_r
                current_c = start_c
                while current_r <= end_r:
                    while current_c <= end_c:
                        min_r = np.int32(current_r - win_size)
                        max_r = np.int32(current_r + win_size)
                        min_c = np.int32(current_c - win_size)
                        max_c = np.int32(current_c + win_size)
                        if (min_r > 0) and (min_c > 0) and (max_r < img_shape[0]) and (max_c < img_shape[1]):
                            data_t = np.reshape(self.img_t[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                            data_tminus1 = np.reshape(self.img_tminus1[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                            data_tminus2 = np.reshape(self.img_tminus2[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                            data_tminus3 = np.reshape(self.img_tminus3[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
                            X = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
                            X = np.expand_dims(X, axis=0)
                            X, _ = bf.DataNormalisationZeroCentred(X, self.aveImg_regression)
                            RegressionResult = self.model_regression.predict(X, batch_size=1, verbose=0)
                            RegressionResult = cv2.resize(np.reshape(RegressionResult, (15, 15)), (45, 45))
                            PredictedPositions_ele_raw.extend(extractPointsFromRegressionImage(RegressionResult, min_r, min_c, T))
                        current_c += win_size
                    current_c = start_c
                    current_r += win_size
            # We only consider the detections with in thebounding box
            PredictedPositions_ele = []
            for d_ele in PredictedPositions_ele_raw:
                if (d_ele[0] > BoundingBox[0]) and (d_ele[0] < BoundingBox[2]) \
                        and (d_ele[1] > BoundingBox[1]) and (d_ele[1] < BoundingBox[3]):
                    PredictedPositions_ele.append(d_ele)

            PredictedPositions_final.extend(PredictedPositions_ele)
        self.PredictedPositions = PredictedPositions_final
        return PredictedPositions_final

    def do_refine_bs(self):
        RefinedCentres = self.refine_centres()
        print("CNN refinement finished...")
        self.associate_centres_bs()
        print("CNN refined regions of detections are associated with background subtraction region of detections...")
        self.predict_moving_obj_locations()
        print("CNN regression subtraction finished...")
        Detections1 = []
        Detections2 = []
        # Deal with the detections that are directly accepted by binary CNN
        for AcceptedDetectionProp_ele in self.AcceptedDetectionProp:
            tmp_detection_struct = {}
            tmp_detection_struct["centre"] = AcceptedDetectionProp_ele.centroid
            tmp_detection_struct["coords"] = AcceptedDetectionProp_ele.coords
            Detections1.append(tmp_detection_struct)
        # Deal with the detections that are predicted by Regression CNN
        for PredictedPositions_ele in self.PredictedPositions:
            tmp_detection_struct = {}
            tmp_detection_struct["centre"] = PredictedPositions_ele
            tmp_rs = np.expand_dims(np.array(range(PredictedPositions_ele[0]-5, PredictedPositions_ele[0]+6)), axis=1)
            tmp_cs = np.expand_dims(np.array(range(PredictedPositions_ele[1]-5, PredictedPositions_ele[1]+6)), axis=1)
            tmp_detection_struct["coords"] = np.concatenate((tmp_rs, tmp_cs), axis=1)
            Detections2.append(tmp_detection_struct)
        return Detections1, Detections2, RefinedCentres


# if A is an elment in B
def ismember(A, B):
    output = False
    for b in B:
        if np.all(A == b):
            output = True
            break
    return output


def extractPointsFromRegressionImage(regressionImage, offset_r, offset_c, T):
    detections = []
    # print(np.max(regressionImage))
    Y = np.max(regressionImage)
    if Y >= T:
        d_r, d_c = np.where(regressionImage == Y)
        for i in range(len(d_r)):
            detections.append(np.array([d_r[i]+offset_r, d_c[i]+offset_c]))
        Y -= 0.1
        while Y > T:
            regressionImage_Bin = regressionImage >= Y
            labels = measure.label(regressionImage_Bin, connectivity=1)
            for labels_i in range(1, np.max(labels)+1):
                block_r, block_c = np.where(labels == labels_i)
                max_ind = np.argmax(regressionImage[block_r, block_c])
                max_r = block_r[max_ind]
                max_c = block_c[max_ind]
                #max_r, max_c = np.unravel_index(max_ind, (45, 45))
                if not ismember([max_r + offset_r, max_c + offset_c], detections):
                    detections.append(np.array([max_r + offset_r, max_c + offset_c]))
            Y -= 0.1
        Y = T
        regressionImage_Bin = regressionImage >= Y
        labels = measure.label(regressionImage_Bin, connectivity=1)
        for labels_i in range(1, np.max(labels)+1):
            block_r, block_c = np.where(labels == labels_i)
            max_ind = np.argmax(regressionImage[block_r, block_c])
            max_r = block_r[max_ind]
            max_c = block_c[max_ind]
            #max_r, max_c = np.unravel_index(max_ind, (45, 45))
            if not ismember([max_r + offset_r, max_c + offset_c], detections):
                detections.append(np.array([max_r + offset_r, max_c + offset_c]))
    return detections
