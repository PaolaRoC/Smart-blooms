import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
from PIL import Image
import torchvision

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(-2,-1))
    label = np.rot90(label, k, axes=(-2,-1) )
    axis = np.random.randint(0, 1)
    image = np.flip(image, axis=axis+1).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes = (-2,-1), order=0, reshape=False, )
    label = ndimage.rotate(label, angle, axes = (-2,-1), order=0, reshape=False, )
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
   
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
           
        _, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32))

        label = torch.from_numpy(label.astype(np.float32))

        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}

        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        if self.split == 'train_dataset2':
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir + '/TRAIN_img', slice_name+'.png')
            data_img = np.array(Image.open(data_path).resize((224,224)))

            data_img = data_img/255.0
            pixel_mean=[0.485, 0.456, 0.406]
            pixel_std=[0.229, 0.224, 0.225]
            normalized_image = (data_img - pixel_mean) / pixel_std

            data_img = np.transpose(normalized_image, (2,0,1))
            mask_path = os.path.join(self.data_dir + '/TRAIN_mask_ids', slice_name+'.png')
            data_label = np.array(Image.open(mask_path).resize((224,224)))
            image = data_img
            label = data_label 
        else:
            vol_name = self.sample_list[idx].strip('\n') 
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir + '/TEST_img', vol_name+'.png')
            data_img = np.array(Image.open(data_path).resize((224,224)))

            data_img = data_img/255.0
            pixel_mean=[0.485, 0.456, 0.406]
            pixel_std=[0.229, 0.224, 0.225]
            normalized_image = (data_img - pixel_mean) / pixel_std

            data_img = np.transpose(normalized_image, (2,0,1))
            mask_path = os.path.join(self.data_dir + '/TEST_mask_ids', vol_name+'.png')
            data_label = np.array(Image.open(mask_path).resize((224,224)))
            image = data_img
            label = data_label

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
