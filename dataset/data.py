import numpy as np
from torchvision import datasets,transforms
from utils.tool import split_images
import os



class idata(object):
    train_trsf=[]
    test_trsf=[]
    common_trsf=[]
    class_order=None

class icifar10(idata):
    use_path=False
    train_trsf=[
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255)
        ]
    test_trsf=[]
    common_trsf=[
         transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ]
    
    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("../data", train_trsf=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("../data", train_trsf=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
    

class icifar100(idata):
    use_path=False
    train_trsf=[
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
        ]
    test_trsf=[transforms.ToTensor()]
    common_trsf=[
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ]
    
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../download", train_trsf=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../download", train_trsf=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
def trans_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_trans(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class icifar224(idata):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.use_path=False
        self.train_trsf=build_trans(True,args)
        self.test_trsf = build_trans(False, args)
        self.common_trsf = [
        ]

        self.class_order = np.arange(100).tolist()
    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../download", train_trsf=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../download", train_trsf=False, download=True)
        self.train_data, self.train_trsf_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNetR(idata):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.use_path=True
        self.train_trsf=build_trans(True,args)
        self.test_trsf=build_trans(False,args)
        self.common_trsf=[]

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "../download/imagenet-r/train"
        test_dir = "../download/imagenet-r/test"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images(train_dset.imgs)
        self.test_data, self.test_targets = split_images(test_dset.imgs)

class iImageNetA(idata):
    use_path=True
    train_trsf=build_trans(True,None)
    test_trsf=build_trans(False,None)
    common_trsf=[]
    class_order=np.arange(100).tolist()

    def download_data(self):
        os.environ['ina_train_dir'] = "C:/Users/sapta/Desktop/ease2/ina/train"
        train_dir = os.environ.get('ina_train_dir')
        os.environ['ina_test_dir'] = "C:/Users/sapta/Desktop/ease2/ina/test"
        test_dir = os.environ.get('ina_test_dir')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images(train_dset.imgs)
        self.test_data, self.test_targets = split_images(test_dset.imgs)