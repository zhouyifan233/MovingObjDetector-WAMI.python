import numpy as np
import tensorflow as tf
import hdf5storage
import glob
import TrainNetwork.BaseFunctions as bf
import pickle
import tensorflow.keras.backend as K

def mean_squared_error(y_true, y_pred):
    return K.mean(K.sum(K.square(y_pred - y_true), axis=1), axis=None)

def mean_absolute_error(y_true, y_pred):
    return K.mean(K.sum(K.abs(y_pred - y_true), axis=1), axis=None)

numData = 1000000
numData_vali = 10000

data_path0 = "E:/WPAFB-data-for-training/Vehicle-Position-Regression/winsize45-multiframes-withbg_1"
data_pathset = glob.glob(data_path0+"/*.mat")
data_all = []
label_all = []
set_cnt = 0
for thispath in data_pathset:
    mat = hdf5storage.loadmat(thispath)
    data_tmp = mat.get("dataset")
    label_tmp = mat.get("labelset")
    data_all.extend(data_tmp)
    label_all.extend(label_tmp)
    set_cnt += 1
    print("Read dataset: " + str(set_cnt))
    if set_cnt == 300:
        break
del data_path0, data_pathset, data_tmp, label_tmp, mat, set_cnt

data_randperm_idx = np.random.permutation(len(data_all))
X = np.zeros((numData, 4, 45, 45), dtype=np.float32)
Y = np.zeros((numData, 225), dtype=np.float32)
for i in range(numData):
    thisidx = data_randperm_idx[i]
    data_t = np.reshape(data_all[thisidx][:, 0], (1, 45, 45))
    data_tminus1 = np.reshape(data_all[thisidx][:, 2], (1, 45, 45))
    data_tminus2 = np.reshape(data_all[thisidx][:, 3], (1, 45, 45))
    data_tminus3 = np.reshape(data_all[thisidx][:, 4], (1, 45, 45))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X[i] = X_tmp
    Y[i] = label_all[thisidx]
print("Generating training data finished...")

X_vali = np.zeros((numData_vali, 4, 45, 45), dtype=np.float32)
Y_vali = np.zeros((numData_vali, 225), dtype=np.float32)
for i in range(numData_vali):
    thisidx = data_randperm_idx[numData+i+1]
    data_t = np.reshape(data_all[thisidx][:, 0], (1, 45, 45))
    data_tminus1 = np.reshape(data_all[thisidx][:, 2], (1, 45, 45))
    data_tminus2 = np.reshape(data_all[thisidx][:, 3], (1, 45, 45))
    data_tminus3 = np.reshape(data_all[thisidx][:, 4], (1, 45, 45))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X_vali[i] = X_tmp
    Y_vali[i] = label_all[thisidx]
print("Generating validation data finished...")

del data_all, data_t, data_tminus1, data_tminus2, data_tminus3,\
    X_tmp, data_randperm_idx

X, averageX = bf.DataNormalisationZeroCentred(X)
X_vali, _ = bf.DataNormalisationZeroCentred(X_vali, averageX)

'''Generate Network'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='valid', input_shape=(4, 45, 45),
                                 data_format='channels_first', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                 bias_initializer='zeros'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', data_format='channels_first',
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), bias_initializer='zeros'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_first'))

model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first',
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), bias_initializer='zeros'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first',
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), bias_initializer='zeros'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), bias_initializer='zeros'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(225))

# tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=False)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9, decay=1e-5, nesterov=False),
              loss=mean_squared_error,
              metrics=[mean_absolute_error])
#model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0),
#              loss='mean_squared_error',
#              metrics=['mae'])

'''Train the network'''
model.fit(X, Y, epochs=100, batch_size=100, validation_data=(X_vali, Y_vali), verbose=1, shuffle=True)

model.save("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/Regression/saved_model_3.model")
writer_fid = open("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/Regression/saved_image_norm_3.model", "wb")
pickle.dump(averageX, writer_fid)
writer_fid.close()

