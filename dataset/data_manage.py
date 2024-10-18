import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.data import icifar10, icifar100, icifar224, iImageNetR,iImageNetA

class DataManager(object):
    def __init__(self,data_name,shuffle,seed,init_cls,inc,args):
        self.args=args
        self.data_name=data_name
        self._setup_data(data_name,shuffle,seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._incs=[init_cls]
        while sum(self._incs)+inc<len(self._class_order):
            self._incs.append(inc)
        offset = len(self._class_order) - sum(self._incs)
        if offset > 0:
            self._incs.append(offset)
    @property
    def nb_task(self):
        return len(self._incs)
    def task_size(self,task):
        return self._incs[task]
    
    
    @property
    def nb_classes(self):
        return len(self._class_order)
    
    def get_dataset(self,ind,source,mode,append=None,ret_data=False,m_rate=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in ind:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if append is not None and len(append) != 0:
            appendent_data, appendent_targets = append
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)
    
    def get_data_split(self,ind,source,mode,append=None,val_sample_per_class=0):
        if source=="train":
            x,y=self._train_data,self._train_targets
        elif source=="test":
            x,y=self._test_data,self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))
        
        if mode=="train":
            trsf=transforms.Compose([*self._train_trsf,*self._common_trsf])
        elif mode=="test":
            trsf=transforms.Compose([*self._test_trsf,*self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in ind:
            class_data,class_targets=self._select(x,y,low_range=idx,high_range=idx+1)
            val_indx = np.random.choice(
                len(class_data), val_sample_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])
        if append is not None:
            append_data, append_targets = append
            for idx in range(0,int(np.max(append_targets))+1):
                app_data,app_targets=self._select(append_data,append_targets,low_range=idx,high_range=idx+1)
                val_indx = np.random.choice(
                    len(app_data), val_sample_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(app_data))))- set(val_indx)
                val_data.append(app_data[val_indx])
                val_targets.append(app_targets[val_indx])
                train_data.append(app_data[train_indx])
                train_targets.append(app_targets[train_indx])
        
        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path),DummyDataset(val_data, val_targets, trsf, self.use_path)


    def _setup_data(self,data_name,shuffle,seed):
        idata = _get_idata(data_name, self.args)
        idata.download_data()
        
        #data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        #transform
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        #order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)


    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
    
    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

class DummyDataset(Dataset):
    def __init__(self,images,labels,trsf,use_path=False):
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.use_path:
            image=self.trsf(pil_loader(self.images[idx]))
        else:
            image=self.trsf(Image.fromarray(self.images[idx]))

        label = self.labels[idx]
        return idx, image, label

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return icifar10()
    elif name == "cifar100":
        return icifar100()
    elif name == "cifar224":
        return icifar224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
     with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


        


