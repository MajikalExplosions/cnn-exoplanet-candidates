import torch
import torch.nn as nn
import torch.optim as optim

import math, random
import numpy as np

from os.path import exists

##Possible ways to reduce mult-adds:
# - Increase pooling for conv layers
# - Reduce spatial data (not in large amounts)
# - Decrease filter count
_test_res = []

def save_res():
    a = np.array(_test_res)
    app = 0
    while exists("res/training_results_" + str(app) + ".npy"):
        app += 1
    np.save("res/training_results_" + str(app), a)
    
def test_model(model, data, batch_size, hide, flip_chance, device, opt, loss, res_index=0):
    print("S")
    np.random.shuffle(data)

    batches_data = [np.zeros((batch_size, 1, 2000, 7, 7))]
    batches_class = [np.zeros((batch_size))]

    #Create batches
    counter = 0
    for i in range(len(data) // 8 * 8):
        row = data[i]
        if hide:
            c = hide_random(np.copy(row[2]))
        else:
            c = np.copy(row[2])

        if random.random() < flip_chance:
            flip_h(c)
        if random.random() < flip_chance:
            flip_v(c)

        batches_data[-1][counter] = c
        batches_class[-1][counter] = row[1]

        counter += 1
        if counter == batch_size:
            batches_data.append(np.empty((batch_size, 1, 2000, 7, 7)))
            batches_class.append(np.empty((batch_size)))
            counter = 0
    
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for i in range(len(batches_data)):
            d = torch.from_numpy(batches_data[i]).to(device)
            c = torch.from_numpy(batches_class[i]).long().to(device)
            #Clear gradients
            opt.zero_grad()

            #Test
            p = model(d.float())

            #Find accuracy
            p2 = torch.argmax(p, dim=1)
            total_correct += len([i for i in range(len(p2)) if p2[i] == c[i]])

            #Calculate loss and store
            l = loss(p, c)
            total_loss += l.item() * batch_size

            del d
            del c
    
    res = [
        total_correct,
        len(batches_data) * batch_size,
        total_correct / (len(batches_data) * batch_size),
        total_loss,
        total_loss / (batch_size * len(batches_data))
    ]
    print("[TEST]  Total Correct: {0} / {1} ({2:.2f}%) | Total Loss: {3:.2f} (Avg. {4:.4f})".format(res[0], res[1], res[2] * 100, res[3], res[4]))
    
    if len(_test_res) <= res_index:
        _test_res.append([])
    _test_res[res_index].append(res)
    return res


#ML Model Defines
class TESSClassifier_5(nn.Module):
    def __init__(self):
        super(TESSClassifier_5, self).__init__()
        #TODO Figure out the model
        self.c3d = nn.Sequential(
            TESSClassifier_5._cnn_layer(1, 2, kernel_size=(7, 1, 1), padding=0, pool=(7, 1, 1)),
            #TESSClassifier_5._cnn_layer(1, 8, kernel_size=(7, 3, 3), pool=(5, 1, 1)),
            TESSClassifier_5._cnn_layer(2, 4, pool=(7, 1, 1)),
            #TESSClassifier_5._cnn_layer(8, 8, pool=(5, 1, 1)),
            TESSClassifier_5._cnn_layer(4, 8),
            TESSClassifier_5._cnn_layer(8, 8, pool=(11, 3, 3))
            #TESSClassifier_5._cnn_layer(16, 8, pool=(7, 3, 3))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 32),
            #nn.LeakyReLU(),
            #nn.Dropout(),
            #nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout()
        )
        self.o = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax()
        )
    
    def forward(self, x):
        y = self.c3d(x)
        y = self.fc(y)
        y = self.o(y)
        return y

    def _cnn_layer(i, o, kernel_size=(3, 3, 3), padding=1, stride=1, pool = None):
        c = nn.Conv3d(i, o, kernel_size=kernel_size, padding=padding, stride=stride)
        r = nn.LeakyReLU()
        if not pool is None:
            p = nn.MaxPool3d(pool, padding=(1, 0, 0))
            return nn.Sequential(c, r, p)
        else:
            return nn.Sequential(c, r)

class TESSClassifier_15(nn.Module):
    def __init__(self):
        super(TESSClassifier_15, self).__init__()
        #TODO Figure out the model
        self.c3d = nn.Sequential(
            TESSClassifier_15._cnn_layer(1, 64, kernel_size=(7, 3, 3), pool=(5, 1, 1)),
            TESSClassifier_15._cnn_layer(64, 128, pool=(5, 1, 1)),
            TESSClassifier_15._cnn_layer(128, 256),
            TESSClassifier_15._cnn_layer(256, 256, pool=(7, 3, 3))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11264, 1024),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.o = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.c3d(x)
        x = self.fc(x)
        return self.o(x)

    def _cnn_layer(i, o, kernel_size=(3, 3, 3), padding=1, stride=1, pool = None):
        c = nn.Conv3d(i, o, kernel_size=kernel_size, padding=padding, stride=stride)
        r = nn.LeakyReLU()
        if not pool is None:
            p = nn.MaxPool3d(pool, padding=(1, 0, 0))
            return nn.Sequential(c, r, p)
        else:
            return nn.Sequential(c, r)

def flip_v(arr):
    l = arr.shape[1]
    flips = l // 2
    l -= 1
    for i in range(flips):
        tmp = arr[:, i, :].copy()
        arr[:, i, :], arr[:, l - i, :] = arr[:, l - i, :], tmp
    return arr

def flip_h(arr):
    l = arr.shape[2]
    flips = l // 2
    l -= 1
    for i in range(flips):
        tmp = arr[:, :, i].copy()
        arr[:, :, i], arr[:, :, l - i] = arr[:, :, l - i], tmp
    return arr


def hide_random(arr, max_hidden = 0.25):
    l0 = math.floor(random.random() * len(arr) * max_hidden)
    p0 = math.floor(random.random() * len(arr))
    for i in range(max(p0 - l0, 0), min(p0 + l0, 2000)):
        arr[i][:] = -1
    return arr