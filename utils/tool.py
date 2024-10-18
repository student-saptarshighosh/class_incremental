import os
import numpy as np
import torch

def count_param(model,train=False):
    if train:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    for i, target in enumerate(targets):
        onehot[i, target] = 1.0
    return onehot

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calc_accuracy(y_pred, y_true):
    return np.around(np.mean(y_pred == y_true) * 100, decimals=2)

def group_accuracy(y_pred, y_true, start, end):
    idxes = np.where((y_true >= start) & (y_true < end))[0]
    return calc_accuracy(y_pred[idxes], y_true[idxes]) if len(idxes) > 0 else 0

def accuracy_func(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."

    all_acc = {}
    all_acc["total"] = calc_accuracy(y_pred, y_true)

    # Grouped accuracy
    max_class = np.max(y_true)
    for class_id in range(0, max_class + 1, increment):
        label =  label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = group_accuracy(y_pred, y_true, class_id, class_id + increment)

    # Old and New accuracy
    all_acc["old"] = calc_accuracy(y_pred[y_true < nb_old], y_true[y_true < nb_old])
    all_acc["new"] = calc_accuracy(y_pred[y_true >= nb_old], y_true[y_true >= nb_old])

    return all_acc

def split_images(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)