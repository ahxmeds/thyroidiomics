#%%
import numpy as np 
import glob
import os 
import pandas as pd 
import SimpleITK as sitk
import sys
import argparse
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
import torch
import os
import glob
import pandas as pd
import numpy as np
import torch.nn as nn
import time
from initialize_train import (
    get_validation_sliding_window_size,
    get_model,
    get_train_valid_test_splits,
    get_valid_transforms,
    get_post_transforms
)
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
#%%
def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)

#%%
def main(args):
    # initialize inference
    network = args.network_name
    experiment_code = f"{network}_loco{args.leave_one_center_out}_trn64_val128_WDL"
    sw_roi_size = get_validation_sliding_window_size(args.inference_patch_size) # get sliding_window inference size for given input patch size
    
    # find the best model for this experiment from the training/validation logs
    # best model is the model with the best validation `Metric` (DSC)
    save_models_dir = os.path.join(RESULTS_FOLDER,'models', experiment_code)
    save_logs_dir = os.path.join(RESULTS_FOLDER,'logs', experiment_code)
    
    # save train and valid logs folder
   
    validlog_fname = os.path.join(save_logs_dir, 'validlog_gpu0.csv')
    validlog = pd.read_csv(validlog_fname)
    best_epoch = 2*(np.argmax(validlog['Metric']) + 1)
    best_metric = np.max(validlog['Metric'])
    print(f"Using the {network} model at epoch={best_epoch} with mean valid DSC = {round(best_metric, 4)}")

    # get the best model and push it to device=cuda:0
    best_model_fname = 'model_ep=' + pad_zeros_at_front(best_epoch, 4) +'.pth'
    model_path = os.path.join(save_models_dir, best_model_fname)
    device = torch.device(f"cuda:0")
    model = get_model(network)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
        
    # initialize the location to save predicted masks
    save_preds_dir = os.path.join(RESULTS_FOLDER, f'predictions')
    os.makedirs(save_preds_dir, exist_ok=True)
    save_preds_dir = os.path.join(save_preds_dir, experiment_code)
    os.makedirs(save_preds_dir, exist_ok=True)

    # get test data (in dictionary format for MONAI dataloader), test_transforms and post_transforms
    _, _, test_data = get_train_valid_test_splits(args.leave_one_center_out)
    test_transforms = get_valid_transforms()
    post_transforms = get_post_transforms(test_transforms, save_preds_dir)
    
    # initalize PyTorch dataset and Dataloader
    dataset_test = Dataset(data=test_data, transform=test_transforms)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        for data in dataloader_test:
            inputs = data['ST'].to(device)
            sw_batch_size = args.sw_bs
            data['Pred'] = sliding_window_inference(inputs, sw_roi_size, sw_batch_size, model)
            data = [post_transforms(i) for i in decollate_batch(data)]


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Thyroid segmentation in scintigraphy images using MONAI-PyTorch')
    parser.add_argument('--network-name', type=str, default='unet', metavar='netname',
                        help='network name for training (default: unet)')
    parser.add_argument('--leave-one-center-out', type=str, default='A', metavar='center',
                        help='leave a center out for testing (default: A)')
    parser.add_argument('--inference-patch-size', type=int, default=128, metavar='inputsize',
                        help='size of cropped input patch for inference (default: 192)')
    parser.add_argument('--num_workers', type=int, default=2, metavar='nw',
                        help='num_workers for train and validation dataloaders (default: 2)')
    parser.add_argument('--sw-bs', type=int, default=2, metavar='sw-bs',
                        help='batchsize for sliding window inference (default=2)')
    args = parser.parse_args()
    
    main(args)