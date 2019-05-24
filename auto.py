#!/usr/bin/env python3

## ____  _ _
##| __ )(_) |__   __ _
##|  _ \| | '_ \ / _` |
##| |_) | | |_) | (_| |
##|____/|_|_.__/ \__,_|

## The program is currently set to run in the fully automatic mode
## that replaces the human in the loop with an auto evaluator.

## This program was run on windows platform and has not been tested on Linux

from bibalabel.automatic import *

if __name__ == '__main__':

    parameters1 = ['kmeans', 72, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters2 = ['kmeans', 424, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters3 = ['kmeans', 257, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters4 = ['kmeans', 1148, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters5 = ['kmeans', 348, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters6 = ['kmeans', 1630, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters7 = ['kmeans', 1903, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters8 = ['kmeans', 844, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters9 = ['kmeans', 32, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters10 = ['kmeans', 1283, 20, 'cosine', 10, 0.997, False, 'combine']


    #   parameters = [method, random_seed, number, distance, iteration, stricktness, kmeansReset, postprocessing]

    #                  method(string): method of selecting initial images or ommit reset step (random, kmeans or none)
    #                           In case "none", "imageList.csv" must be prepared manualy

    #                  random_seed(int): Random seed for image selection. Random seed for training is defiend in "CONSTANTS.py"

    #                  number(int): initial number of labeled images that starts the first iteration

    #                  distance(string): distance base for kmeans calculation (euclidean or cosine)

    #                  iteration(int): the whole "training, prediction, evaluation" loop will repeat this much

    #                  stricktness(float): minumum acceptable dice value for a good prediction 0.00-1.00

    #                  kmeansReset(Bool): if True, feature space of a Resnet50 will be buillt before calculating the kmeans
    #                                     if False, kmeans uses previous feature space in the models\ directory


    #                  postprocessing(string): "crf", "contour", "combine", "combine2", "combine3", "convexHull" or "none"
    #                            combine = "crf" | "contour"
    #                            combine2 = "crf" & "contour"
    #                            combine3 = ("crf" & "contour") | "contour"



    autoPilot(parameters1)
    autoPilot(parameters2)
    autoPilot(parameters3)
    autoPilot(parameters4)
    autoPilot(parameters5)
    autoPilot(parameters6)
    autoPilot(parameters7)
    autoPilot(parameters8)
    autoPilot(parameters9)
    autoPilot(parameters10)
