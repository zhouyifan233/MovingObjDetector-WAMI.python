import numpy as np
import tensorflow as tf
import hdf5storage
import glob
import TrainNetwork.BaseFunctions as bf
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
import pickle
import matplotlib.pyplot as plt
import cv2
import os

model = tf.keras.models.load_model("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/Regression/saved_model_3.model")
reader_fid = open("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/Regression/saved_image_norm_3.model", "rb")
averageX = pickle.load(reader_fid)
reader_fid.close()

imagefolder = "E:/WAMI-test/WASABI-AngelFire-test/"
num_of_template = 5
IDX = 14

# Read image folder
filenames = os.listdir(imagefolder)
filenames.sort()
print(str(len(filenames)) + " images in the folder...")
# Load background
images = []
for i in range(num_of_template):
    ReadImage = cv2.imread(imagefolder + filenames[i+IDX], cv2.IMREAD_GRAYSCALE)
    images.append(ReadImage)
bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

input_image = cv2.imread(imagefolder + filenames[num_of_template+IDX], cv2.IMREAD_GRAYSCALE)
print("Filename: " + filenames[i+IDX])
Hs = bgt.doCalculateHomography(input_image)
CompensatedImages = bgt.doMotionCompensationAndValidArea(input_image, Hs, input_image.shape)

plt.figure()
plt.imshow(input_image)
plt.show()

img_t = input_image
img_tminus1 = CompensatedImages[num_of_template-1]
img_tminus2 = CompensatedImages[num_of_template-2]
img_tminus3 = CompensatedImages[num_of_template-3]

tr = 1175
tc = 799

win_size = 22
img_shape = input_image.shape
min_r = np.int32(tr - win_size)
max_r = np.int32(tr + win_size)
min_c = np.int32(tc - win_size)
max_c = np.int32(tc + win_size)
if (min_r > 0) and (min_c > 0) and (max_r < img_shape[0]) and (max_c < img_shape[1]):
    data_t = np.reshape(img_t[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
    data_tminus1 = np.reshape(img_tminus1[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
    data_tminus2 = np.reshape(img_tminus2[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
    data_tminus3 = np.reshape(img_tminus3[min_r:max_r+1, min_c:max_c+1], (1, 45, 45))
    X = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X = np.expand_dims(X, axis=0)
    X, _ = bf.DataNormalisationZeroCentred(X, averageX)
    RegressionResult = model.predict(X, batch_size=1, verbose=0)
    RegressionResult = cv2.resize(np.reshape(RegressionResult, (15, 15)), (45, 45))

plt.figure()
plt.imshow(RegressionResult)
plt.show()

plt.figure()
plt.imshow(img_t[min_r:max_r+1, min_c:max_c+1])
plt.show()

plt.figure()
plt.imshow(img_tminus1[min_r:max_r+1, min_c:max_c+1])
plt.show()

plt.figure()
plt.imshow(img_tminus2[min_r:max_r+1, min_c:max_c+1])
plt.show()

plt.figure()
plt.imshow(img_tminus3[min_r:max_r+1, min_c:max_c+1])
plt.show()

