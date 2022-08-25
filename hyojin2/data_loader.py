import random
from random import Random
import constant as const

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as dset
import torchvision.transforms as T


def get_train_validation_loader():
    train_dataset = dset.ImageFolder(root='./flatfish_train')
    val_dataset = dset.ImageFolder(root='./flatfish_val')

    train_dataset = FlatfishTrain(train_dataset, num_train=const.NUM_TRAIN)
    # val_dataset = FlatfishTrain(val_dataset, num_train=const.NUM_VAL)
    val_dataset = FlatfishTest(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=const.BATCH_SIZE, shuffle=True)

    return train_loader, val_loader

def get_test_loader():
    test_dataset = dset.ImageFolder(root='./flatfish_test')

    test_dataset = FlatfishTest(test_dataset)
    # test_dataset = FlatfishTrain(test_dataset, num_train=const.NUM_TEST)

    test_loader = DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=True)

    return test_loader

class FlatfishTrain(Dataset):
    def __init__(self, dataset, num_train):
        self.dataset = dataset
        self.num_train = num_train

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        # Get image from same class
        #   : 홀수번째 데이터는 같은 클래스의 이미지
        if index % 2 == 1:
            # 같으면 Label=1
            label = 1.0
            # ['sea', 'farm'] 중 랜덤으로 하나 선택 (0:farm / 1:sea)
            idx = random.randint(0, len(self.dataset.classes) - 1)
            # 선택한 class의 이미지 데이터(주소) 리스트
            image_list = [x for x in self.dataset.imgs if x[1] == idx]
            # 선택한 class의 이미지 리스트 중 2개의 이미지 선택
            image1 = random.choice(image_list)
            image2 = random.choice(image_list)
            # *2개의 이미지가 같으면 두번째 이미지를 다시 선택
            while image1[0] == image2[0]:
                image2 = random.choice(image_list)

        # Get image from different class
        #   : 짝수번째 데이터는 다른 클래스의 이미지
        else:
            # 다르면 Label=0
            label = 0.0
            # 랜덤으로 2개의 이미지 선택
            image1 = random.choice(self.dataset.imgs)
            image2 = random.choice(self.dataset.imgs)
            # *2개의 이미지의 레이블이 같으면 두번째 이미지를 다시 선택
            while image1[1] == image2[1]:
                image2 = random.choice(self.dataset.imgs)

        # 이미지를 넘파이 형태로 로드 후 Tensor로 변환 및 1@105x105사이즈로 resize
        image1 = cv2.imread(image1[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image2[0], cv2.IMREAD_GRAYSCALE)
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        image1 = torch.tensor(image1).resize(1, 105, 105).type(torch.FloatTensor)
        image2 = torch.tensor(image2).resize(1, 105, 105).type(torch.FloatTensor)
        label = torch.tensor(label)

        return image1, image2, label

class FlatfishTest:
    def __init__(self, dataset, trials=2, way=4, seed=0):
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.seed = seed
        self.num_test = len(dataset)*2
        # self.image1 = None

    def __len__(self):
        # return self.trials * self.way
        return self.num_test

    def __getitem__(self, index):
        rand = Random(self.seed + index)
        # get image pair from same class
        if index < len(self.dataset):
            image1 = self.dataset.imgs[0]
            image2 = self.dataset.imgs[index]
        else:
            image1 = self.dataset.imgs[5]
            image2 = self.dataset.imgs[index - len(self.dataset)]

        if image1[1]==image2[1]:
            label = 1.0
        else:
            label = 0.0

        # if index % self.way == 0:
        #     label = 1.0
        #     idx = rand.randint(0, len(self.dataset.classes) - 1)
        #     image_list = [x for x in self.dataset.imgs if x[1] == idx]
        #     self.image1 = rand.choice(image_list)
        #     image2 = rand.choice(image_list)
        #     while self.image1[0] == image2[0]:
        #         image2 = rand.choice(image_list)

        # # get image pair from different class
        # else:
        #     label = 0.0
        #     image2 = random.choice(self.dataset.imgs)
        #     while self.image1[1] == image2[1]:
        #         image2 = random.choice(self.dataset.imgs)
        
        image1 = cv2.imread(image1[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image2[0], cv2.IMREAD_GRAYSCALE)
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        image1 = torch.tensor(image1).resize(1, 105, 105).type(torch.FloatTensor)
        image2 = torch.tensor(image2).resize(1, 105, 105).type(torch.FloatTensor)
        label = torch.tensor(label)

        return image1, image2, label

if __name__=='__main__':
    # train, val = get_train_validation_loader()
    # test = get_test_loader()

    # print(len(train.dataset))
    # print(len(val.dataset))
    # print(len(test.dataset))

    # print(train.dataset.dataset.classes)

    # val_dataset = dset.ImageFolder(root='/Users/hyojin/Fish_Siamese/hyojin2/flatfish_val')
    # val_dataset = FlatfishTrain(val_dataset, num_train=25)
    # img = cv2.imread(train_dataset.imgs[0][0], cv2.IMREAD_GRAYSCALE)
    # img = torch.tensor(img).resize(1, 105, 105)
    # print(img)
    # print(val_dataset[-1])
    testloader = get_test_loader()
    print(len(testloader.dataset))