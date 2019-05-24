#!/usr/bin/python3
import cv2
import csv
import glob
import shutil
import os
import numpy as np
from .CONSTANTS import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
from PIL import Image,ImageTk
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


def makeImageList(Flag=0):
    imageList = []
    for filename in glob.glob(os.path.join(IMAGE_DN,'*.JPG')):
        imageList.append([filename[6:],'N'])
        if os.path.isfile(os.path.join(MASK_DN,filename[6:-3])+'PNG') or Flag:
            imageList[-1][1]=1
    return imageList

def readImageList():
    try:
        with open(IMAGE_FN, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            imageList = list(reader)
    except:
        imageList=[]
    if len(imageList):
        imageList = imageList[1:]
    else:
        imageList = makeImageList()
    return imageList

def readEasyList(easyName):
    try:
        with open(os.path.join(TMP_PATH,easyName)+'.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            imageList = list(reader)
            imageList = imageList[1:]
    except:
        imageList=[]
        for filename in glob.glob(os.path.join(IMAGE_DN, '*.JPG')):
            imageList.append([filename[6:]])
    return imageList

def badImageList(imageList):
    resultList=[]
    for item in imageList:
        if item[1]==1 or item[1]=='1':
            resultList.append(item[0][:-4])
    return resultList

def goodImageList(imageList):
    resultList=[]
    for item in imageList:
        if item[1]==0 or item[1]=='0':
            resultList.append(item[0][:-4])
    return resultList

def notGoodImageList(imageList):
    resultList=[]
    for item in imageList:
        if item[1]!='0' and item[1]!=0:
            resultList.append(item)
    return resultList

def delAllMask():
    folder = MASK_DN
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def writeImageList(imageList, backup = 1):
    if backup ==1:
        listName = IMAGE_FN
    else:
        listName = os.path.join(TMP_PATH,IMAGE_FN)
    
    with open(listName, 'w+', newline='') as f:
        imageList.insert(0,['image', 'status'])
        wr = csv.writer(f, delimiter=',')
        wr.writerows(imageList)


def overlay(mask, clr =(255,255,255), width = SIZE, height=SIZE):

    Flag = 2*len(mask.shape)-3
    contour = np.zeros((width, height, Flag), np.uint8)

    _, thresh = cv2.threshold(mask[:, :, 2], 100, 255, 0) if Flag ==3 else cv2.threshold(mask, 100, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        # find Convex Hull of the contour
        hull = cv2.convexHull(c)
        cv2.fillPoly(contour, pts=[c], color=clr)
        contour = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        indices = np.where(contour == 255)
        contour[indices[0], indices[1], :] = clr


    return contour

def imageOverlay(imageName, clr =(0,0,255),
                 width = SIZE, height=SIZE, save = 0):

    #reads image and it's mask (if existed)
    #then extracts Convex Hull of the largest contour on mask
    #and returns a combined image
    image = cv2.imread(os.path.join(IMAGE_DN,imageName))
    image = cv2.resize(image,(width,height))

    try:
        mask = cv2.imread(os.path.join(MASK_DN, imageName[:-3]) + 'PNG')
        mask = cv2.resize(mask, (width, height))

    except:
        mask = np.zeros((width, height, 3), np.uint8)

    contour = overlay(mask, clr, width, height)

    if (save): return contour
    dst = cv2.addWeighted(image,0.7,contour,0.3,0)
    return dst

def readOverlay(imageName, color =(0,0,255), width =SIZE, height =SIZE):
    im = imageOverlay(imageName, color, width, height)
    return imageRearrange(im)

def imageRearrange(img):
    #Rearrang the color channel
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    return imgtk

def cpyFile(name):
#    if os.path.isfile(os.path.join(GROUNDT_DN,name[:-3])+'PNG'):
    name2 = name[:-3]+'PNG'
    shutil.copyfile(os.path.join(GROUNDT_DN,name2), os.path.join(MASK_DN,name2))

def maskSave(imageName, draft = 0):
    img = imageOverlay(imageName,save= 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 0, 255, 0)
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    add = TMP_PATH if draft else MASK_DN
    cv2.imwrite(os.path.join(add,imageName[:-3])+'PNG',img)

def postSave(name):
    name2 = name+'.PNG'
    shutil.copyfile(os.path.join(POSTPROCESS_DN,name2), os.path.join(MASK_DN,name2))

def maskDraftDelete(imageName):
    try:
        os.remove(os.path.join(TMP_PATH,imageName[:-3])+'PNG')
    except:
        pass
    
def maskDraftMove():
    for i, filename in enumerate(glob.glob(os.path.join(TMP_PATH,'*.PNG'))):
        shutil.move(filename, os.path.join(MASK_DN,filename[4:]))


def report(string):
    if string=='now':
        import datetime
        now = datetime.datetime.now()
        string = f'\n\n\n{now}'
    with open("report.txt", "a") as textFile:
        print(string, file = textFile)


def set_vector():
    # 0. define model and normalization
    model = models.resnet50(pretrained=True)
    layer = model._modules.get('avgpool')

    # Set model to evaluation mode
    model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    return model, layer, scaler, normalize, to_tensor


def get_vector(image_name, model, layer, scaler, normalize, to_tensor):
    # 1. Load the image with Pillow library
    img = Image.open(os.path.join(IMAGE_DN,image_name))
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 2048 for resnet50
    my_embedding = torch.zeros(2048)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


def makeDataArray(imageList):
    import sys
    imageData = np.array([], dtype=np.uint8).reshape(0,2048)
    n=len(imageList)
    model, layer, scaler, normalize, to_tensor = set_vector()
    print('Building Resnet50 features for all images...')
    for i, item in enumerate(imageList):
      vector = get_vector(item, model, layer, scaler, normalize, to_tensor)
      A = np.asarray(vector)
      imageData = np.vstack([imageData, A])
      toolbar_width = int(40*(i+1)/n)
      sys.stdout.write("\r (%d%%) [%s]" % (int(2.5*toolbar_width), "█" * toolbar_width))
      sys.stdout.flush()
    return imageData


def clustering(image_data, numberOfClusters, seed = 0, distance= 'euclidean', init = 'k-means++', n_init = 10):
    if seed!=0: np.random.seed(seed)
    if distance =='cosine':
        x = preprocessing.normalize(image_data)
    elif distance =='euclidean':
        x = image_data
    else:
        raise Exception('Error! distance of the kmens function must be either "euclidean" or "cosine".')
    k = KMeans(n_clusters=numberOfClusters, random_state=RANDOMSEED, init =init, n_init= n_init).fit(x)
    labels = []
    results = []
    for i in range(numberOfClusters):
        labels.append(np.argwhere(k.labels_ == i).flatten()) #find all images in cluster number i, make a 1d list out of it
        a = np.random.choice(labels[i], 1) #randomly choose one item from this list
        results.append(a[0]) #add the selected image to the list of final results
    return results


def ordinal(n):
    # converts number to ordinal number
    result = "%d%s" % (n, "tsnrhtdd"[(n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
    return  result

if __name__ == "__main__":
    a = readImageList()
    writeImageList(a)
