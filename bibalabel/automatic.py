from .bladerTools import *
from .CONSTANTS import *
import random
import time
from .train import *
import sys
from .statisticalMeasurements import *
from .postProcess import *
from shutil import copyfile

def postCopy(name, postID):
    name2 = name+'.PNG'
    folder = POSTPROCESS_DN if postID==2 else POSTPROCESS2_DN
    copyfile(os.path.join(folder,name2), os.path.join(MASK_DN,name2))


def autoEvaluate(easyName, iteration, strictness=STRICTNESS):

    imageList = readImageList()
    easyList = readEasyList(easyName)
    itemNo = 0
    ln = len(imageList[0])
    n = len(imageList)
    print("\n\n----------------Auto Evaluation------------------------")
    for i, item in enumerate(imageList):
        if item[1] == '1':
            dice1 = diceCoefficient(item[0][:-4],2)
            dice2 = diceCoefficient(item[0][:-4],3)
            dice = max(dice1,dice2)
            diceID = 2 if dice == dice1 else 3
        else:
            dice = 1.01
        if ln > 2:
            imageList[i][2] = dice
        else:
            imageList[i].append(dice)

        easyList[i].append(dice)

        if (item[1] == '1' or item[1] == 1) and dice >= strictness:

            itemNo += 1
            postSave(item[0][:-4])
            postCopy(item[0][:-4], diceID)
            imageList[i][1] = '0'

        toolbar_width = int(20 * (i + 1) / n)
        sys.stdout.write(f"\r({int(5 * toolbar_width)})% [{'█' * toolbar_width}]")
        sys.stdout.flush()

    print(f"with strictness of {strictness*100}, {itemNo} images are added as good")
    report("----------------Auto Evaluation------------------------")
    report(f"with strictness of {strictness*100}, images are added as good: {itemNo}")
    writeImageList(imageList)
    report(f'images are modified')
    report('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    header = ['image', 'status']
    for i in range(iteration):
        header.append(ordinal(i+1)+' iteration')
    with open(os.path.join(TMP_PATH, easyName)+'.csv', 'w+', newline='') as f:
        easyList.insert(0, header)
        wr = csv.writer(f, delimiter=',')
        wr.writerows(easyList)


def reset(method='random', seed = 0, number=20, distance='euclidean', kmeansReset=False):
    #This function deletes all the predicted masks from mask folder.
    #Then Selects a group of images and copies the relevant mask from the GT folder

    easyName = time.strftime("%Y%m%d-%H%M%S") #name of a temp csv file that keeps record of all evaluations
    report('now')               #writes on report.text file
    report(70 * "-")
    report(f'Random Seed = {seed}')
    report(f'Initial Number of Images = {number}')
    if os.path.isfile('models/512urn.h5'): os.unlink('models/512urn.h5')
    theList = makeImageList(Flag=1)     #makes a new list of files

    if method == 'random':
        delAllMask()  # deletes all mask from previous predictions
        report('Random initialization')
        if seed != 0: random.seed(int(seed))        #if seed is zero, selection will be without a fixed seed
        ra = random.sample(range(1, len(theList) + 1), int(number))
        for a in ra:
            theList[a - 1][1] = '0'
            cpyFile(theList[a - 1][0])

    elif method == 'kmeans':
        delAllMask()  # deletes all mask from previous predictions
        print('K-means initialization')
        report('K-means initialization')
        report(f'Distance = {distance}')
        if kmeansReset:
            # read only the name of the images and
            # make an array of a Resnet50 feature space and
            # save it in the models\ directory
            report('start making kmeans')
            report('now')
            myin = makeDataArray(next(zip(*theList)))
            np.save('models\\imagedataForKmeans', myin)
            report('finish making kmeans')
            report('now')
        myin = np.load('models\imagedataForKmeans.npy')
        report('start clustering')
        kList = clustering(myin, int(number), seed, distance)
        for i in kList:
            theList[i][1] = '0'
            cpyFile(theList[i][0])
        report('finished clustering')
        report('now')
    elif method == 'none':
        shutil.copyfile(IMAGE_FN, os.path.join(TMP_PATH, easyName) + '.csv')
        return easyName
    else:
        raise Exception('Error! method of the reset function must be either "random", "kmeans" or "none".')

    writeImageList(theList)
    shutil.copyfile(IMAGE_FN, os.path.join(TMP_PATH, easyName) + '.csv')
    return easyName


def trainPredictEvaluate(iteration, postMethod, stricktness, easyname):
    report(f'Number of Iterations: = {iteration}')
    postList = ["crf", "contour", "convexHull", "crf", "contour", "convexHull", "crf", "contour", "convexHull", "crf", "contour", "convexHull"]
    for i in range (1,iteration+1):
        #postMethod = postList[i-1]
        print(f'\n\n{5*"-"}{ordinal(i)} Iteration{45*"-"}\n')
        report(f'\n\n{5*"-"}{ordinal(i)} Iteration{35*"-"}\n')
        random_seed(RANDOMSEED, True)

        training()
        predictImages()
        postProcessing(postMethod)
        autoEvaluate(easyname, iteration, stricktness)

    report('now')
    report(f'{10*"-"}FINISHED{10*"-"}')
    report(70*"*")


def autoPilot(parameters, resume = False):
    #   parameters = [method, random_seed, number, distance, iteration, stricktness, kmeansReset, postprocessing]
    #                  method(string): method of selecting initial images or ommit reset step (random, kmeans or none)
    #                  random_seed(int): Random seed for image selection. Random seed for training is defiend in CONSTANTS
    #                  number(int): initial number of labeled images that starts the first training
    #                  distance(string): distance base for kmeans calculation (euclidean or cosine)
    #                  iteration(int): the whole "training, prediction, evaluation" loop will repeat this much
    #                  stricktness(float): minumum dice value for a good prediction 0.00-1.00
    #                  kmeansReset(Bool): if True, feature space of a Resnet50 will be buillt before calculating the kmeans
    #                                     if False, kmeans uses previous feature space in the models\ directory
    #                  postprocessing(string): "crf", "contour", "combine", "combine2", "combine3", "convexHull" or "none"
    if resume:
        try:
            with open(os.path.join(TMP_PATH, 'easyName.txt'), "r") as f:
                easyName = f.read(15)
        except:
            easyName = time.strftime("%Y%m%d-%H%M%S")
    else:
        easyName = reset(parameters[0], parameters[1], parameters[2], parameters[3], parameters[6])
    trainPredictEvaluate(parameters[4], parameters [7], parameters[5], easyName)
    with open(os.path.join(TMP_PATH, 'easyName.txt'), "w") as f:
        f.write(easyName)

if __name__ == "__main__":
    # reset('random', 72, 20, 'cosine', False, 'k-means++', 10)
    # train.random_seed(RANDOMSEED, True)
    #
    # train.training()
    # train.predictImages()
    autoEvaluate('easyName.txt', 1, strictness=STRICTNESS)
