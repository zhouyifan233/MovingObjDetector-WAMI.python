import numpy as np
import tensorflow as tf
import hdf5storage
import glob
import TrainNetwork.BaseFunctions as bf
import pickle

numPositive = 600000
numNegative = 3000000
numPositive_test = 6000
numNegative_test = 30000

traindata_path0 = "E:\WPAFB-data-for-training\Vehicle-Binary-Classifier\winsize10_1"
traindata_pathset = glob.glob(traindata_path0+"\*.mat")
positive_data_all = []
negative_data_all = []
set_cnt = 0
for thispath in traindata_pathset:
    mat = hdf5storage.loadmat(thispath)
    positive_data_tmp = mat.get("positive_dataset")
    negative_data_tmp = mat.get("negative_dataset")
    positive_data_all.extend(positive_data_tmp)
    negative_data_all.extend(negative_data_tmp)
    set_cnt += 1
    print("Read dataset: " + str(set_cnt))
    if set_cnt == 300:
        break
del traindata_path0, traindata_pathset, positive_data_tmp, negative_data_tmp, mat, set_cnt

positive_data_randperm_idx = np.random.permutation(len(positive_data_all))
negative_data_randperm_idx = np.random.permutation(len(negative_data_all))

X = np.ndarray((numPositive + numNegative, 4, 21, 21), dtype=np.float32)
Y = np.zeros((numPositive + numNegative, 1), dtype=np.uint8)
for i in range(numPositive):
    thisidx = positive_data_randperm_idx[i]
    data_t = np.reshape(positive_data_all[thisidx][:, 0], (1, 21, 21))
    data_tminus1 = np.reshape(positive_data_all[thisidx][:, 1], (1, 21, 21))
    data_tminus2 = np.reshape(positive_data_all[thisidx][:, 2], (1, 21, 21))
    data_tminus3 = np.reshape(positive_data_all[thisidx][:, 3], (1, 21, 21))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X[i] = X_tmp
    Y[i] = 0

for i in range(numNegative):
    thisidx = negative_data_randperm_idx[i]
    data_t = np.reshape(negative_data_all[thisidx][:, 0], (1, 21, 21))
    data_tminus1 = np.reshape(negative_data_all[thisidx][:, 1], (1, 21, 21))
    data_tminus2 = np.reshape(negative_data_all[thisidx][:, 2], (1, 21, 21))
    data_tminus3 = np.reshape(negative_data_all[thisidx][:, 3], (1, 21, 21))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X[i+numPositive] = X_tmp
    Y[i+numPositive] = 1
print("Generating training data finished...")

X_vali = np.ndarray((numPositive_test + numNegative_test, 4, 21, 21), dtype=np.float32)
Y_vali = np.zeros((numPositive_test + numNegative_test, 1), dtype=np.uint8)
for i in range(numPositive_test):
    thisidx = positive_data_randperm_idx[numPositive+i+1]
    data_t = np.reshape(positive_data_all[thisidx][:, 0], (1, 21, 21))
    data_tminus1 = np.reshape(positive_data_all[thisidx][:, 1], (1, 21, 21))
    data_tminus2 = np.reshape(positive_data_all[thisidx][:, 2], (1, 21, 21))
    data_tminus3 = np.reshape(positive_data_all[thisidx][:, 3], (1, 21, 21))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X_vali[i] = X_tmp
    Y_vali[i] = 0

for i in range(numNegative_test):
    thisidx = negative_data_randperm_idx[numNegative+i+1]
    data_t = np.reshape(negative_data_all[thisidx][:, 0], (1, 21, 21))
    data_tminus1 = np.reshape(negative_data_all[thisidx][:, 1], (1, 21, 21))
    data_tminus2 = np.reshape(negative_data_all[thisidx][:, 2], (1, 21, 21))
    data_tminus3 = np.reshape(negative_data_all[thisidx][:, 3], (1, 21, 21))
    X_tmp = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
    X_vali[i + numPositive_test] = X_tmp
    Y_vali[i + numPositive_test] = 1

print("Generating validation data finished...")
del positive_data_all, negative_data_all, data_t, data_tminus1, data_tminus2, data_tminus3,\
    X_tmp, positive_data_randperm_idx, negative_data_randperm_idx

X, averageX = bf.DataNormalisationZeroCentred(X)
X_vali, _ = bf.DataNormalisationZeroCentred(X_vali, averageX)

Y = tf.keras.utils.to_categorical(Y, 2)
Y_vali = tf.keras.utils.to_categorical(Y_vali, 2)

'''Generate Network'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=(4, 21, 21), data_format='channels_first'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_first'))

model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-4, nesterov=False),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''Train the network'''
model.fit(X, Y, epochs=80, batch_size=100, validation_data=(X_vali, Y_vali), verbose=1, shuffle=True)

model.save('C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/BinaryClassification/saved_model_2.model')
writer_fid = open("C:/Users/yifan/Google Drive/PythonSync/MovingObjDetector-WAMI.python/Models/BinaryClassification/saved_image_norm_2.model", "wb")
pickle.dump(averageX, writer_fid)
writer_fid.close()

