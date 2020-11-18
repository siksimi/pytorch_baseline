import os
import cv2
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split

def seed_everything(seed=1234567):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
VAL_RATIO = 0.01
RANDOM_SEED = 0
seed_everything(RANDOM_SEED)
IMSIZE = 256, 256

TRAIN_PATH = 'train'
TEST_PATH = 'test'
OUTPUT_PATH = 'output/'

df_train = pd.read_csv("train.tsv", delimiter='\t', header=None)
df_test = pd.read_csv("test.tsv", delimiter='\t', header=None)

keys ={68:0, 83:1, 86:2, 91:3, 95:4, 113:5, 148:6, 167:7, 174:8, 177:9, 230:10, 245:11, 274:12, 279:13, 282:14, 288:15, 289:16, 290:17, 291:18, 293:19}

train_filelist = df_train.iloc[:,0].values.tolist()
df_train[3] = df_train[1]*21 + df_train[2]
train_y = []
for i in range(len(df_train)):
    val = df_train.iloc[i, 3]
    train_y.append(keys[val])

test_filelist = df_test.iloc[:,0].values.tolist()

train_img = []
test_img = []
print('Loading', len(train_filelist), 'training images ...')
for i, p in enumerate(train_filelist):
    train_img.append(cv2.imread(os.path.join(TRAIN_PATH, p), 1))
print('Loading', len(test_filelist), 'test images ...')
for i, p in enumerate(test_filelist):
    test_img.append(cv2.imread(os.path.join(TEST_PATH, p), 1))

def ImagePreprocessing(img):
    h, w = IMSIZE
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        img[i] = tmp
    print(len(img), 'images processed!')
    return img

def save_model(model_name, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join('./model/R50-1-1/' + model_name + '.pth'))
    print('model saved')

def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join('./model/R50-1-1/' + model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

nb_epoch, batch_size, learning_rate = 20, 40, 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'   

#model = SampleModelTorch()
#model = EfficientNet_b4()

model_name = "efficientnet-b4"
image_size = EfficientNet.get_image_size(model_name)
#print(image_size)
#model = EfficientNet.from_pretrained(model_name, num_classes=20, advprop=False)
model = EfficientNet.from_name(model_name, num_classes=20) # model only
#model = models.resnet50(pretrained=True)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 20)
model_save = '18'
load_model(model_save, model)

print(model)

model.to(device)
criterion1 = nn.CrossEntropyLoss()
#criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('Training start ...')
train_img = ImagePreprocessing(train_img)
train_img = np.array(train_img, dtype=np.float32)
train_y = np.array(train_y)

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), 
    #transforms.RandomResizedCrop((224, 224), scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=2),
    #transforms.RandomRotation(30, resample=False, expand=False, center=None, fill=None),
    #transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0), ratio=(0.8, 1.25), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tfms_val = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])     

tfms_test = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class nipaDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0]
        y = self.tensors[1]
        x = self.transform(x[index])
        return x, y[index]

    def __len__(self):
        return self.tensors[0].size(0)


class nipaTestset(Dataset):
    def __init__(self, x, transform=None):
        self.x = x
        self.transform = transform

    def __getitem__(self, index):
        x = self.x
        x = self.transform(x[index])
        return x

    def __len__(self):
        return self.x.size(0)

img_tr, img_val, y_tr, y_val = train_test_split(train_img, train_y, stratify = train_y, test_size = VAL_RATIO, shuffle=True, random_state=RANDOM_SEED)
tr_set = nipaDataset(torch.from_numpy(img_tr).float(), torch.from_numpy(y_tr).long(), transform=tfms)
val_set = nipaDataset(torch.from_numpy(img_val).float(), torch.from_numpy(y_val).long(), transform=tfms_val)
batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
batch_val = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)

#####   Training loop   #####
STEP_SIZE_TRAIN = len(img_tr) // batch_size
print('\n\n STEP_SIZE_TRAIN= {}\n\n'.format(STEP_SIZE_TRAIN))

t0 = time.time()
for epoch in range(nb_epoch):
    t1 = time.time()
    print('Model fitting ...')
    print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
    print('check point = {}'.format(epoch))
    a1, a1_val, tp1, tp1_val = 0, 0, 0, 0
    pooled_loss, pooled_loss_val = 0, 0
    model.train()
    for i, (x, y) in enumerate(batch_train):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred1 = model(x)
        loss = criterion1(pred1, y)
        pooled_loss += loss
        loss.backward()
        optimizer.step()
        prob1, pred1_cls = torch.max(pred1, 1)
        a1 += y.size(0)
        tp1 += (pred1_cls == y).sum().item()

    model.eval()

    with torch.no_grad():
        for j, (x, y) in enumerate(batch_val):
            x, y = x.to(device), y.to(device)
            pred_val = model(x)
            loss_val = criterion1(pred_val, y)
            pooled_loss_val += loss_val
            prob1_val, pred1_cls_val = torch.max(pred_val, 1)
            a1_val += y.size(0)
            tp1_val += (pred1_cls_val == y).sum().item()

    acc1 = tp1 / a1
    acc1_val = tp1_val / a1_val
    print("  * loss = {}\n  * acc = {}\n  * loss_val = {}\n  * acc_val = {}\n".format(loss.item(), acc1, 
                                                                                    loss_val.item(), acc1_val))
    print("  * loss1 = {}\n  * loss1v = {}\n".format(pooled_loss, pooled_loss_val))
    print('Training time for one epoch : %.1f\n' % (time.time() - t1))
    save_model(str(epoch + 1), model, optimizer)
    
print('Total training time : %.1f' % (time.time() - t0))


# inference with test set 
print('Test start ...')

test_img = ImagePreprocessing(test_img)
test_img = np.array(test_img, dtype=np.float32)
test_dataset = nipaTestset(torch.from_numpy(test_img).float(), transform=tfms_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)


model_save = '9'
load_model(model_save, model)

model.eval()
pred_test = []
with torch.no_grad():
    for j, x in enumerate(test_dataloader):
        x = x.to(device)
        pred = model(x)
        prob, pred_cls = torch.max(pred, 1)
        pred_test += pred_cls.tolist()
print('Prediction done!\n Saving the result...')
df_output = pd.DataFrame(list(zip(test_filelist, pred_test)))
rev_keys ={0:68, 1:83, 2:86, 3:91, 4:95, 5:113, 6:148, 7:167, 8:174, 9:177, 10:230, 11:245, 12:274, 13:279, 14:282, 15:288, 16:289, 17:290, 18:291, 19:293}
for i in range(len(df_output)):
    val = df_output.iloc[i, 1]
    df_output.iloc[i, 1] = rev_keys[val]
df_output[2] = df_output[1] % 21
df_output[1] = df_output[1] //21
df_output.to_csv(os.path.join(OUTPUT_PATH, 'r50-{}.tsv'.format(time.time())), sep = '\t', index=False, header=None)