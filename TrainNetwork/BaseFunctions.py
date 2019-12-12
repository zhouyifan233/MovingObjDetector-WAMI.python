import numpy as np
import pickle
import tensorflow as tf

def DataNormalisationZeroCentred(InputData, AverageData=None):

    if AverageData is None:
        AverageData = np.mean(InputData, axis=0)
        NormalisedData = InputData - AverageData
    else:
        NormalisedData = InputData - AverageData

    return NormalisedData, AverageData


def ReadModels(model_folder):

    model_binary = tf.keras.models.load_model(
        model_folder + "/BinaryClassification/saved_model_2.model")
    reader_fid = open(
        model_folder + "/BinaryClassification/saved_image_norm_2.model", "rb")
    aveImg_binary = pickle.load(reader_fid)
    reader_fid.close()

    model_regression = tf.keras.models.load_model(
        model_folder + "/Regression/saved_model_3.model")
    reader_fid = open(
        model_folder + "/Regression/saved_image_norm_3.model", "rb")
    aveImg_regression = pickle.load(reader_fid)
    reader_fid.close()

    return model_binary, aveImg_binary, model_regression, aveImg_regression

