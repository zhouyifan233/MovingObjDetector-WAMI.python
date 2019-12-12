import numpy as np
import tensorflow as tf
import hdf5storage
import glob
import TrainNetwork.BaseFunctions as bf
import pickle
import timeit

model = tf.keras.models.load_model("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/Regression/saved_model_2.model")
reader_fid = open("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.pythonn/Models/Regression/saved_image_norm.model_2", "rb")
averageX = pickle.load(reader_fid)
reader_fid.close()

data_path0 = "E:/WPAFB-data-for-training/Vehicle-Position-Regression/winsize45-multiframes-withbg"
data_pathset = glob.glob(data_path0+"/*.mat")
data_all = []
label_all = []
set_cnt = 0
for idx, thispath in enumerate(data_pathset):
    if idx > 300:
        mat = hdf5storage.loadmat(thispath)
        data_tmp = mat.get("dataset")
        label_tmp = mat.get("labelset")
        data_all.extend(data_tmp)
        label_all.extend(label_tmp)
        set_cnt += 1
        print("Read dataset: " + str(set_cnt))
del data_path0, data_pathset, data_tmp, label_tmp, mat, set_cnt

numData = len(data_all)
X_test = np.zeros((numData, 4, 45, 45), dtype=np.float32)
Y_test = np.zeros((numData, 225), dtype=np.float32)
for i in range(numData):
    data_t = np.reshape(data_all[i][:, 0], (1, 45, 45))
    data_tminus1 = np.reshape(data_all[i][:, 2], (1, 45, 45))
    data_tminus2 = np.reshape(data_all[i][:, 3], (1, 45, 45))
    data_tminus3 = np.reshape(data_all[i][:, 4], (1, 45, 45))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X_test[i] = X_tmp
    Y_test[i] = label_all[i]
print("Generating testing data finished...")
del data_all, label_all, data_t, data_tminus1, data_tminus2, data_tminus3,\
    X_tmp
X_test, _ = bf.DataNormalisationZeroCentred(X_test, averageX)

starttime = timeit.default_timer()
predictResults = model.predict(X_test, batch_size=5000, verbose=1)
endtime = timeit.default_timer()
print("Processing Time (CNN prediction): " + str(endtime - starttime) + " s... ")

SE = []
for idx in range(len(predictResults)):
    thisError = predictResults[idx] - Y_test[idx]
    thisError = np.sum(np.square(thisError))
    SE.append(thisError)
MSE = np.mean(SE)

