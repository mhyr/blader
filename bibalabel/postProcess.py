from .convcrf import do_crf_inference
from .bladerTools import *
import sys
import time


def preCrf(name):
    # unary is the output of the last layer of the network before thresholding
    unaryName = os.path.join(MASK_DN, name+'.npy')
    img = cv2.imread(os.path.join(IMAGE_DN, name+'.jpg'))
    a1 = np.load(unaryName)
    b = (a1[0] - np.min(a1[0])) / np.ptp(a1[0])
    c = 1 - b
    unary = np.dstack((c, b))
    prediction = do_crf_inference(img, unary, True)
    return prediction


def preCrf2(name):
    # unary is the binary mask
    img = cv2.imread(os.path.join(IMAGE_DN, name+'.'+EXTENSION))
    mask = cv2.imread(os.path.join(MASK_DN, name+'.png'), 0)
    b = (mask/255).astype(np.float)
    c = 1 - b
    unary = np.dstack((c, b))
    prediction = do_crf_inference(img, unary)
    return prediction


def contour(name, width=SIZE, height=SIZE):
    mask = cv2.imread(os.path.join(MASK_DN, name+'.png'), cv2.IMREAD_GRAYSCALE)

    canvas = np.zeros((width, height), np.uint8)  # define an empty canvas to draw the contour on it
    _, thresh = cv2.threshold(mask, 128, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        cv2.fillPoly(canvas, pts=[c], color=(255, 255, 255))
    else:
        return None
    return c, canvas

def contour2(mask, width=SIZE, height=SIZE):
    canvas = np.zeros((width, height), np.uint8)  # define an empty canvas to draw the contour on it
    _, thresh = cv2.threshold(mask, 128, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        cv2.fillPoly(canvas, pts=[c], color=(255, 255, 255))
    else:
        return None
    return c, canvas


def hull(name, width=SIZE, height=SIZE):
    canvas = np.zeros((width, height), np.uint8)  # define an empty canvas to draw the contour on it
    c, _ = contour(name, width=width, height=height)
    convex = cv2.convexHull(c)
    cv2.fillPoly(canvas, pts=[convex], color=(255, 255, 255))
    return canvas


def postProcess(name, method='crf', save=True):
    if method == 'crf':
        result2 = preCrf2(name)
        result = result2

    elif method == 'combine':
        result2 = preCrf2(name)
        _, result = contour(name)

    elif method == 'combine2':
        holder = preCrf2(name)
        _, result2 = contour2(holder)
        result = result2
    elif method == 'combine3':
        holder = preCrf2(name)
        _, result2 = contour2(holder)
        _, result = contour(name)

    elif method == 'contour':
        _, result = contour(name)
        result2 = result

    elif method == 'convexHull':
        result = hull(name)
        result2 = result

    elif method == 'none':
        result = cv2.imread(os.path.join(MASK_DN, name+'.png'), cv2.IMREAD_GRAYSCALE)
        result2 = result
    else:
        print('method must be "crf", "contour", "combine", "combine2", "combine3", "convexHull" or "none"!')
        return None

    if save:
        cv2.imwrite(os.path.join(POSTPROCESS_DN, name+'.png'), result)
        cv2.imwrite(os.path.join(POSTPROCESS2_DN, name + '.png'), result2)
    return result2


def postProcessing(method='combine'):
    start = time.time()
    imageList = readImageList()
    n = len(imageList)
    print("----------------post processing------------------------")
    for i, item in enumerate(imageList):
        if item[1] == '1':
            postProcess(item[0][:-4], method)

        toolbar_width = int(20 * (i + 1) / n)
        sys.stdout.write(f"\r({int(5 * toolbar_width)})% [{'█' * toolbar_width}]")
        sys.stdout.flush()

    report(f'----------------post processing with {method}----------------')
    print(f"Post processing time: {time.time()-start}(Sec)")
    report(f"Post processing: {time.time()-start}(Sec)\n")


if __name__ == "__main__":
    image = 'DSC_2418_1'
    a = preCrf(image)
    cv2.imshow(image, a)
    cv2.waitKey(0)
