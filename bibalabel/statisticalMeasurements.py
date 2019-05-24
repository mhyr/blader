import cv2
import os
from .CONSTANTS import *
import numpy as np


def falsePositive(item, postID):
    folder = POSTPROCESS_DN if postID == 2 else POSTPROCESS2_DN
    if postID == 4: folder = MASK_DN
    prediction = cv2.imread(os.path.join(folder,item)+'.png', cv2.IMREAD_GRAYSCALE)
    groundTruth = cv2.imread(os.path.join(GROUNDT_DN,item)+'.png', cv2.IMREAD_GRAYSCALE)
    fpImage = cv2.subtract(prediction, groundTruth)
    result = np.sum(fpImage == 255)
    return result


def falseNegative(item, postID):
    folder = POSTPROCESS_DN if postID == 2 else POSTPROCESS2_DN
    if postID == 4: folder = MASK_DN
    prediction = cv2.imread(os.path.join(folder,item)+'.png', cv2.IMREAD_GRAYSCALE)
    groundTruth = cv2.imread(os.path.join(GROUNDT_DN,item)+'.png', cv2.IMREAD_GRAYSCALE)
    fnImage = cv2.subtract(groundTruth, prediction)
    result = np.sum(fnImage == 255)
    return result


def truePositive(item, postID):
    folder = POSTPROCESS_DN if postID == 2 else POSTPROCESS2_DN
    if postID == 4: folder = MASK_DN
    prediction = cv2.imread(os.path.join(folder,item)+'.png', cv2.IMREAD_GRAYSCALE)
    groundTruth = cv2.imread(os.path.join(GROUNDT_DN,item)+'.png', cv2.IMREAD_GRAYSCALE)
    tpImage = cv2.bitwise_and(prediction, groundTruth)
    result = np.sum(tpImage == 255)
    return result


def trueNegative(item, postID):
    folder = POSTPROCESS_DN if postID == 2 else POSTPROCESS2_DN
    if postID == 4: folder = MASK_DN
    prediction = cv2.imread(os.path.join(folder,item)+'.png', cv2.IMREAD_GRAYSCALE)
    prediction = cv2.bitwise_not(prediction)
    groundTruth = cv2.imread(os.path.join(GROUNDT_DN,item)+'.png', cv2.IMREAD_GRAYSCALE)
    groundTruth = cv2.bitwise_not(groundTruth)
    tnImage = cv2.bitwise_and(prediction, groundTruth)
    result = np.sum(tnImage == 255)
    return result


def accuracy(item, postID):
    tp = truePositive(item, postID)
    tn = trueNegative(item, postID)
    fp = falsePositive(item, postID)
    fn = falseNegative(item, postID)
    result = (tp+tn)/(tp+tn+fp+fn)
    return result


def sensitivity(item, postID):
    tp = truePositive(item, postID)
    fn = falseNegative(item, postID)
    result = tp/(tp+fn)
    return result


def specificity(item, postID):
    tn = trueNegative(item, postID)
    fp = falsePositive(item, postID)
    result = tn/(tn+fp)
    return result


def percision(item, postID):
    tp = truePositive(item, postID)
    fp = falsePositive(item, postID)
    result = tp/(tp+fp)
    return result


def diceCoefficient(item, postID):
    tp = truePositive(item, postID)
    fp = falsePositive(item, postID)
    fn = falseNegative(item, postID)
    result = (2*tp)/(2*tp+fp+fn)
    return result
