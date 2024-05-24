'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
#%%
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Spacingd,
    RandAffined,
    ScaleIntensityRanged,
    Invertd,
    AsDiscreted,
    SaveImaged,
    RandCropByPosNegLabeld,
    SpatialCropd,
    
)
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import torch
import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys 
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import DATA_FOLDER, WORKING_FOLDER
from sklearn.model_selection import train_test_split
#%%
def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)
 
def create_dictionary_stgt(stpaths, gtpaths):
    data = [{'ST':stpath, 'GT':gtpath} for stpath, gtpath in zip(stpaths, gtpaths)]
    return data

def remove_all_extensions(filename):
    while True:
        name, ext = os.path.splitext(filename)
        if ext == '':
            return name
        filename = name

def get_train_valid_test_splits_onlyonecenter(center='A'):
    datapath = '/home/jhubadmin/Projects/thyroid-segmentation/segmentation/datainfo_images_to_use.csv'
    data = pd.read_csv(datapath)
    data = data[data['Class'] != 0] # remove normal class 
    ids = data[data['CenterID'] == center]['PatientID'].tolist()
    stpaths = [os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in ids]
    gtpaths = [os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in ids]
    data_dict = create_dictionary_stgt(stpaths, gtpaths)
    train_data, test_data = train_test_split(data_dict, test_size=0.2)
    train_data, valid_data = train_test_split(train_data, test_size=0.2)
    return train_data, valid_data, test_data
    
def get_train_valid_test_splits(leave_one_center_out='A'):
    datapath = '/home/jhubadmin/Projects/thyroid-segmentation/segmentation/datainfo_images_to_use.csv'
    data = pd.read_csv(datapath)
    data = data[data['Class'] != 0] # remove normal class 
    trainvalid_ids = data[data['CenterID'] != leave_one_center_out]['PatientID'].tolist()
    test_ids = data[data['CenterID'] == leave_one_center_out]['PatientID'].tolist()

    stpaths_trainvalid = [os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in trainvalid_ids]
    gtpaths_trainvalid = [os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in trainvalid_ids]

    stpaths_test = [os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in test_ids]
    gtpaths_test = [os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in test_ids]

    trainvalid_data = create_dictionary_stgt(stpaths_trainvalid, gtpaths_trainvalid)
    test_data = create_dictionary_stgt(stpaths_test, gtpaths_test)
    train_data, valid_data = train_test_split(trainvalid_data, test_size=0.2, random_state=42)
    return train_data, valid_data, test_data

#%%

def get_spatial_size(input_patch_size=64):
    trsz = input_patch_size
    return (trsz, trsz)

def get_spacing():
    spc = 1
    return (spc, spc)

def get_train_transforms(input_patch_size=64):
    spatialsize = get_spatial_size(input_patch_size)
    spacing = get_spacing()
    mod_keys = ['ST', 'GT']
    train_transforms = Compose(
    [
        LoadImaged(keys=mod_keys, image_only=True),
        EnsureChannelFirstd(keys=mod_keys),
        ScaleIntensityRanged(keys=['ST'], a_min=0, a_max=550, b_min=0, b_max=1, clip=True),
        Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'nearest')),
        RandCropByPosNegLabeld(
            keys=mod_keys,
            label_key='GT',
            spatial_size=spatialsize,
            pos=2,
            neg=1
        ),
        RandAffined(
            keys=mod_keys,
            mode=('bilinear', 'nearest'),
            prob=0.5,
            spatial_size = spatialsize,
            translate_range=(5,5),
            rotate_range=[np.pi/12],
            scale_range=(0.1, 0.1)),
    ])

    return train_transforms

#%%
def get_valid_transforms():
    spacing = get_spacing()
    mod_keys = ['ST', 'GT']
    valid_transforms = Compose(
    [
        LoadImaged(keys=mod_keys, image_only=True),
        EnsureChannelFirstd(keys=mod_keys),
        ScaleIntensityRanged(keys=['ST'], a_min=0, a_max=550, b_min=0, b_max=1, clip=True),
        Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'nearest')),
    ])

    return valid_transforms


def get_post_transforms(test_transforms, save_preds_dir):
    post_transforms = Compose([
        Invertd(
            keys="Pred",
            transform=test_transforms,
            orig_keys="GT",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="Pred", argmax=True),
        SaveImaged(keys="Pred", meta_keys="pred_meta_dict", output_dir=save_preds_dir, output_postfix="", separate_folder=False, resample=False),
    ])
    return post_transforms

def get_kernels_strides(patch_size, spacings):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    sizes, spacings = patch_size, spacings
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides
#%%
def get_model(network_name = 'unet'):
    if network_name == 'unet':
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    if network_name == 'unet1':
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    if network_name == 'unet2':
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            norm=Norm.BATCH
        )
    if network_name == 'unet3':
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    return model


#%%
class WeightedDiceLoss(nn.Module):
    def __init__(self, weight_fp=1.0):
        super(WeightedDiceLoss, self).__init__()
        self.weight_fp = weight_fp
    
    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs[:, 1, :, :]

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        false_positives = (inputs * (1 - targets)).sum()
        weighted_dice_loss = 1 - dice + self.weight_fp * (false_positives / (union + smooth))
        return weighted_dice_loss

def get_loss_function():
    loss_function = WeightedDiceLoss(weight_fp=2)
    return loss_function

def get_optimizer(model, learning_rate=2e-4, weight_decay=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def get_metric():
    metric = DiceMetric(include_background=False, reduction="mean")
    return metric

def get_scheduler(optimizer, max_epochs=500):
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)
    return scheduler

def get_validation_sliding_window_size(inference_patch_size=128):
    windowsize = get_spatial_size(inference_patch_size)
    return windowsize
