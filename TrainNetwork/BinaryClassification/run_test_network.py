import numpy as np
import tensorflow as tf
import hdf5storage
import glob
import TrainNetwork.BaseFunctions as bf
import pickle
import timeit

model = tf.keras.models.load_model('C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/BinaryClassification/saved_model.model')
reader_fid = open("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/BinaryClassification/saved_image_norm.model", "rb")
averageX = pickle.load(reader_fid)
reader_fid.close()

traindata_path0 = "E:\WPAFB-data-for-training\Vehicle-Binary-Classifier\winsize10"
traindata_pathset = glob.glob(traindata_path0+"\*.mat")
positive_data_all = []
negative_data_all = []
set_cnt = 0
for idx, thispath in enumerate(traindata_pathset):
    if idx > 300:
        mat = hdf5storage.loadmat(thispath)
        positive_data_tmp = mat.get("positive_dataset")
        negative_data_tmp = mat.get("negative_dataset")
        positive_data_all.extend(positive_data_tmp)
        negative_data_all.extend(negative_data_tmp)
        set_cnt += 1
        print("Read dataset: " + str(set_cnt))

del traindata_path0, traindata_pathset, positive_data_tmp, negative_data_tmp, mat, set_cnt

numPositive = len(positive_data_all)
numNegative = len(negative_data_all)
X_test = np.ndarray((numPositive + numNegative, 4, 21, 21), dtype=np.float32)
Y_test = np.zeros((numPositive + numNegative, 1), dtype=np.uint8)

for i in range(numPositive):
    data_t = np.reshape(positive_data_all[i][:, 0], (1, 21, 21))
    data_tminus1 = np.reshape(positive_data_all[i][:, 1], (1, 21, 21))
    data_tminus2 = np.reshape(positive_data_all[i][:, 2], (1, 21, 21))
    data_tminus3 = np.reshape(positive_data_all[i][:, 3], (1, 21, 21))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X_test[i] = X_tmp
    Y_test[i] = 0

for i in range(numNegative):
    data_t = np.reshape(negative_data_all[i][:, 0], (1, 21, 21))
    data_tminus1 = np.reshape(negative_data_all[i][:, 1], (1, 21, 21))
    data_tminus2 = np.reshape(negative_data_all[i][:, 2], (1, 21, 21))
    data_tminus3 = np.reshape(negative_data_all[i][:, 3], (1, 21, 21))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X_test[i+numPositive] = X_tmp
    Y_test[i+numPositive] = 1
print("Generating testing data finished...")

del positive_data_all, negative_data_all, data_t, data_tminus1, data_tminus2, data_tminus3,\
    X_tmp

X_test, _ = bf.DataNormalisationZeroCentred(X_test, averageX)
Y_test = tf.keras.utils.to_categorical(Y_test, 2)
#TestLoss = model.evaluate(X_test, Y_test, batch_size=10000, verbose=1)
starttime = timeit.default_timer()
predictResults = model.predict(X_test, batch_size=5000, verbose=1)
endtime = timeit.default_timer()
print("Processing Time (CNN prediction): " + str(endtime - starttime) + " s... ")

AccurateCnt = 0
for idx in range(len(predictResults)):
    thisResult = np.round(predictResults[idx])
    thisLabel = Y_test[idx]
    if all(thisResult == thisLabel):
        AccurateCnt += 1
therate = AccurateCnt/len(predictResults)
print("Accuracy" + str(therate))