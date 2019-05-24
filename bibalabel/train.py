# coding: utf-8
'''
Main source Fastai tutorials 2018
https://github.com/fastai/fastai/blob/master/courses/dl2/carvana-unet.ipynb
'''
from fastai.conv_learner import *
from fastai.dataset import *
from .bladerTools import report
from .CONSTANTS import *

PATH = Path(PATH)
torch.cuda.set_device(0)

def random_seed(seed_value, use_cuda):
    # To ensure reproduciblility, we have to define a fixed seed for
    # torch’s random functions as well as python random functions
    # we also had to set num_workers=0 (or 0) in the DataLoader
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def read_titles(path, image_list_file,
                image_folder, mask_folder,
                Delimiter=',', good=0, bad=1):

    image_csv = pd.read_csv(path/image_list_file, delimiter = Delimiter, usecols=[0,1])
    x_names = np.array([Path(image_folder)/o for o in image_csv['image'].loc[image_csv['status'] == good]])
    y_names = np.array([Path(mask_folder)/f'{o[:-4]}.png' for o in image_csv['image'].loc[image_csv['status'] == good]])
    test_names = np.array([path/Path(image_folder)/o for o in image_csv['image'].loc[image_csv['status'] == bad]])
    result_names = np.array([path/Path(mask_folder)/f'{o[:-4]}.png' for o in image_csv['image'].loc[image_csv['status'] == bad]])
    # test_names = np.array([path/Path(image_folder)/o for o in image_csv['image'].loc[image_csv['status'] == good]])
    # result_names = np.array([path/Path(mask_folder)/f'{o[:-4]}.png' for o in image_csv['image'].loc[image_csv['status'] == good]])
    
    return x_names, y_names, test_names, result_names


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0


def get_base(f, cut):
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)


def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


class SaveFeatures():
    features=None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = UnetBlock(256,3,16)
        self.up6 = nn.ConvTranspose2d(16, 1, 1)
        
    def forward(self,x):
        inp = x
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x, inp)
        x = self.up6(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()


class UnetModel():
    def __init__(self, lr_cut, model,name='unet'):
        self.model,self.name = model,name
        self.lr_cut = lr_cut

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [self.lr_cut]))
        return lgs + [children(self.model)[1:]]


def lerner():
    x_names, y_names, test_names, result_names = read_titles(PATH, IMAGE_FN, IMAGE_DN, MASK_DN)

    split_size = int(x_names.size / 4)

    val_idxs = list(random.sample(range(x_names.size), split_size))

    ((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, x_names, y_names)

    aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
                RandomFlip(tfm_y=TfmType.CLASS),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]

    tfms = tfms_from_model(resnet34, SIZE, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
    datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
    md = ImageData(PATH, datasets, BATCHSIZE, num_workers=0, classes=None)

    f = resnet34
    cut, lr_cut = model_meta[f]

    m_base = get_base(f, cut)
    m = to_gpu(Unet34(m_base))
    models = UnetModel(lr_cut, m)

    learn = ConvLearner(md, models, tmp_name=TMP_PATH, models_name=MODELS_DN)
    learn.opt_fn = optim.Adam
    learn.crit = nn.BCEWithLogitsLoss()
    learn.metrics = [accuracy_thresh(0.5), dice]
    # learn.load('fixedweights') #To be deterministic, however setting all the random seeds, did the trick. so no need.


    return learn, x_names.size, test_names, result_names


def predictImages():
    # a function to do the prediction of the remaining images
    start = time.time()
    learn, _, test_names, result_names = lerner()
    trn_tfms, val_tfms = tfms_from_model(learn.load('512urn'), SIZE)

    n = len(test_names)
    print(f"\npredicting {n} images...")
    for i, file in enumerate(test_names):
        img = open_image(file)
        im = val_tfms(img)
        preds = learn.predict_array(im[None])
        # t = str(result_names[i])
        # np.save(t[:-4], preds)
        mask = (preds[0] > 0)*255
        cv2.imwrite(str(result_names[i]), mask)

        toolbar_width = int(40*(i+1)/n)
        sys.stdout.write("\r (%d%%) [%s]" % (int(2.5 * toolbar_width), "█" * toolbar_width))
        sys.stdout.flush()

    print("\n\n-------Prediction is finished--------------------------")
    report("\n-------Prediction is finished--------------------------")
    report(f"Number of the predicted images: {n}")
    print(f"Prediction time: {time.time()-start}(Sec)")
    report(f"prediction time: {time.time()-start}(Sec)\n")


def training():
    # main training function
    start = time.time()
    lr = 2e-2
    wd = 1e-7
    learn, n, _, _ = lerner()
    lrs = np.array([lr/200, lr/20, lr])/2
    # try:
    #      learn.load('512urn')
    #      report("model is loaded")
    # except:
    #      pass

    learn.freeze_to(1)
    learn.fit(lr, 1, wds=wd, cycle_len=5, use_clr=(5, 5))
    learn.unfreeze()
    learn.bn_freeze(True)
    learn.fit(lrs/2, 1, wds=wd, cycle_len=8, use_clr=(20, 8))
    learn.save('512urn')

    print('\n-------Training is Finished----------------------------')
    report('-------Training is Finished----------------------------')
    print(f"trained images: {n}")
    report(f"trained images: {n}")
    print(f"training time: {time.time()-start}(Sec)")
    report(f"training time: {time.time()-start}(Sec)")


if __name__ == "__main__":
    random_seed(RANDOMSEED, True)
    predictImages()
