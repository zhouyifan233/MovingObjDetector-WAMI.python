import cv2
import numpy as np
import MovingObjectDetector.ImageProcFunc as ImageProcessing
import matplotlib.pyplot as plt
import skimage.measure as measure
import skimage.filters as filters


class BackgroundModel:

    def __init__(self, num_of_template, templates):
        self.num_of_templates = num_of_template
        self.templates = templates
        self.motion_matrices = np.ndarray(shape=[3, 3, self.num_of_templates], dtype=np.float32)
        self.CompensatedImages = []
        self.background = []
        self.invalidArea = []
        self.Hs = None
        self.Background = []

    def showTemplate(self):
        for i in range(self.num_of_templates):
            plt.figure(i)
            plt.imshow(np.repeat(np.expand_dims(self.templates[i], -1), 3, axis=2))
            plt.show()

    def showCompensatedImages(self):
        for idx, cimg in enumerate(self.CompensatedImages):
            plt.figure(idx)
            plt.imshow(np.repeat(np.expand_dims(cimg, -1), 3, axis=2))
            plt.show()

    def getTemplates(self):
        return self.templates

    def getCompensatedImages(self):
        return self.CompensatedImages

    def updateTemplate(self, new_image):
        num_of_templates = self.num_of_templates
        self.templates[0:num_of_templates-1] = self.templates[1:num_of_templates]
        self.templates[num_of_templates-1] = new_image
        self.Hs[0:num_of_templates-1] = self.Hs[1:num_of_templates]
        self.Hs[num_of_templates-1] = []
        return

    def doBackgroundSubtraction(self, input_image, thres=10, CompensateBrightness=True):
        if CompensateBrightness:
            for i, thisTemplate in enumerate(self.CompensatedImages):
                diff = np.float64(input_image) - np.float64(thisTemplate)
                diff = cv2.GaussianBlur(diff, (21, 21), sigmaX=8)
                self.CompensatedImages[i] = np.uint8(thisTemplate + diff)
        thisBackground = np.median(self.CompensatedImages, axis=0)
        self.Background = thisBackground
        subtractionResult = np.abs(input_image - thisBackground)
        subtractionResultBW = np.uint8(subtractionResult >= thres)
        subtractionResultBW[self.invalidArea] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        subtractionResultBW = cv2.morphologyEx(subtractionResultBW, cv2.MORPH_OPEN, kernel)
        subtractionResultBW_ds = cv2.resize(subtractionResultBW, (int(subtractionResultBW.shape[1]/3), int(subtractionResultBW.shape[0]/3)), interpolation=cv2.INTER_NEAREST)
        r, c = np.where(subtractionResultBW_ds == 1)
        r = np.int32(3 * (r-1) + 2)
        c = np.int32(3 * (c-1) + 2)
        CandiateRegionCentres = np.array(list(zip(r, c)))
        BackgroundSubtractionLabels = measure.label(subtractionResultBW, connectivity=1)
        BackgroundSubtractionProperties = measure.regionprops(BackgroundSubtractionLabels)
        # centres = []
        # for thisProperty in Properties:
        #    centres.append([thisProperty.centroid[1], thisProperty.centroid[0]])
        # centres = np.round(np.asarray(centres))
        return CandiateRegionCentres, BackgroundSubtractionProperties, BackgroundSubtractionLabels

    def doMotionCompensationAndValidArea(self, input_image, motion_matrix, dstShape):
        self.CompensatedImages = []
        validAreaAll = np.ones(dstShape, dtype=bool)
        for idx, srcImage in enumerate(self.templates):
            CompensatedImage = ImageProcessing.ImageRegistration(srcImage, dstShape, motion_matrix[idx])
            TmpValidArea = (CompensatedImage == 255) | (CompensatedImage == 0)
            CalcValidArea = measure.label(TmpValidArea, connectivity=2)
            CalcValidAreaProp = measure.regionprops(CalcValidArea)
            for thisProperty in CalcValidAreaProp:
                if thisProperty.area > 10000:
                    Coords = thisProperty.coords
                    validAreaAll[Coords[:, 0], Coords[:, 1]] = False
            self.CompensatedImages.append(CompensatedImage)
        TmpValidArea = (input_image == 255) | (input_image == 0)
        CalcValidArea = measure.label(TmpValidArea, connectivity=2)
        CalcValidAreaProp = measure.regionprops(CalcValidArea)
        for thisProperty in CalcValidAreaProp:
            if thisProperty.area > 10000:
                Coords = thisProperty.coords
                validAreaAll[Coords[:, 0], Coords[:, 1]] = False
        validAreaAll = np.uint8(validAreaAll)
        validAreaAll = cv2.erode(validAreaAll, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))
        validAreaAll = np.bool_(validAreaAll)
        self.invalidArea = np.logical_not(validAreaAll)
        return self.CompensatedImages

    def doCalculateHomography(self, dst_image):
        Hs = []
        if not self.Hs:
            for srcImage in self.templates:
                H, _ = ImageProcessing.CalcHomography(srcImage, dst_image, num_of_features=5000)
                Hs.append(H)
        else:
            H, _ = ImageProcessing.CalcHomography(self.templates[self.num_of_templates-1], dst_image, num_of_features=5000)
            for idx in range(self.num_of_templates-1):
                Hs.append(np.matmul(H, self.Hs[idx]))
            Hs.append(H)
        self.Hs = Hs
        return Hs

    def doUpdateHomography(self, MatricesSet, frame_idx):
        Hs = []
        if not self.Hs:
            Hs_reverse = []
            Hs_tmp = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            for i in range(self.num_of_templates):
                Hs_tmp = np.matmul(Hs_tmp, MatricesSet[frame_idx-1-i][0])
                Hs_reverse.append(Hs_tmp)
            Hs = [Hs_reverse[i] for i in range(self.num_of_templates-1, -1, -1)]
        else:
            for i in range(self.num_of_templates - 1):
                Hs.append(np.matmul(MatricesSet[frame_idx-1][0], self.Hs[i]))
            Hs.append(MatricesSet[frame_idx-1][0])
        self.Hs = Hs
        return Hs
