#!/usr/bin/env python3
## This repository contains the reference implementation for my
## master thesis entiteled " Semi Automatic Data Labeling for
## Deep Learning Applications: A Case Study on Wind Turbine Blade Segmentation"
## at Bremen and Bremer Institut für Produktion und Logistik GmbH (BIBA)
## Examiners: Prof. Dr.-Ing. Walter Lang, Prof. Dr.-Ing. Michael Freitag
## Supervisors: Benjamin Staar and Dimitri Denhof

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

    parameters1 = ['random', 72, 20, 'cosine', 10, 0.997, False, 'none']
    parameters2 = ['random', 424, 20, 'cosine', 10, 0.997, False, 'combine']
    parameters3 = ['random', 257, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters4 = ['random', 1148, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters5 = ['random', 348, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters6 = ['random', 1630, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters7 = ['random', 1903, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters8 = ['random', 844, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters9 = ['random', 32, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']
    parameters10 = ['random', 1283, 20, 'cosine', 10, 0.997, False, 'k-means++', 10, 'combine']


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
    #autoPilot(parameters2)
    #autoPilot(parameters3)
    #autoPilot(parameters4)
    #autoPilot(parameters5)
    #autoPilot(parameters6)
    #autoPilot(parameters7)
    #autoPilot(parameters8)
    #autoPilot(parameters9)
    #autoPilot(parameters10)
