import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import Config
import pickle
import torch
import os
import math
import random
import numpy as np
import time

# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12,10),
                          cmap=plt.cm.Blues,
                          path='result.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)

# Timer
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def ToTensor(list, is_len=False):
    np_ts = np.array(list)
    tensor = torch.from_numpy(np_ts).long()

    if is_len:
        mat1 = np.equal(np_ts, 0)
        mat2 = np.equal(mat1, False)
        lens = np.sum(mat2, axis=1)
        return tensor, lens

    return tensor

def ToAudioLens(list):
    
    np_ts = np.array(list)
    np_ts = np_ts * 1000
    tensor = torch.from_numpy(np_ts).long()
    mat1 = np.equal(np_ts, Config.PAD)
    mat2 = np.equal(mat1, False)
    lens = np.sum(mat2, axis=1)

    return lens

# model saver
def model_saver(model, path, dataset):
    if not os.path.isdir(path):
        os.makedirs(path)
    model_path = '{}/{}.pt'.format(path, dataset)
    torch.save(model, model_path)

# model loader
def model_loader(path, dataset):
    model_path = '{}/{}.pt'.format(path, dataset)
    model = torch.load(model_path, map_location='cpu')
    return model


def saveToJson(path, object):
    t = json.dumps(object, indent=4)
    f = open(path, 'w')
    f.write(t)
    f.close()

    return 1


def saveToPickle(path, object):
    file = open(path, 'wb')
    pickle.dump(object, file)
    file.close()

    return 1


def loadFrPickle(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj



def shuffle_lists(featllist, labellist=None, thirdparty=None):

    if labellist == None:
        random.shuffle(featllist)
        return featllist
    elif labellist != None and thirdparty == None:
        combined = list(zip(featllist, labellist))
        random.shuffle(combined)
        featllist, labellist = zip(*combined)
        return featllist, labellist
    else:
        combined = list(zip(featllist, labellist, thirdparty))
        random.shuffle(combined)
        featllist, labellist, thirdparty = zip(*combined)
        return featllist, labellist, thirdparty

# clipping could be done by Pytorch function: torch.nn.utils.clip_grad_norm_
def param_clip(model, optimizer, batch_size, max_norm=10):
    # gradient clipping
    shrink_factor = 1
    total_norm = 0

    for p in model.parameters():
        if p.requires_grad:
            p.grad.data.div_(batch_size)
            total_norm += p.grad.data.norm() ** 2
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        # print("Total norm of grads {}".format(total_norm))
        shrink_factor = max_norm / total_norm
    current_lr = optimizer.param_groups[0]['lr']

    return current_lr, shrink_factor