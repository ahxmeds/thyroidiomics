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
from config import SEGMENTATION_RESULTS_FOLDER, WORKING_FOLDER, DATA_FOLDER
import matplotlib.pyplot as plt 
#%%

def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)

def read_image_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T

def calculate_patient_level_dice_score(
    gtarray: np.ndarray,
    predarray: np.ndarray, 
) -> np.float64:
    dice_score = 2.0*np.sum(predarray[gtarray == 1])/(np.sum(gtarray) + np.sum(predarray))
    return dice_score

#%%
def main(args):
    # initialize inference
    network = args.network_name
    experiment_code = f"{network}_loco{args.leave_one_center_out}"
    sw_roi_size = get_validation_sliding_window_size(args.inference_patch_size) # get sliding_window inference size for given input patch size
    
    # find the best model for this experiment from the training/validation logs
    # best model is the model with the best validation `Metric` (DSC)
    save_models_dir = os.path.join(SEGMENTATION_RESULTS_FOLDER,'models', experiment_code)
    save_logs_dir = os.path.join(SEGMENTATION_RESULTS_FOLDER,'logs', experiment_code)
    
    # save train and valid logs folder
   
    validlog_fname = os.path.join(save_logs_dir, 'validlog_gpu0.csv')
    validlog = pd.read_csv(validlog_fname)
    best_epoch = args.val_interval*(np.argmax(validlog['Metric']) + 1)
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
    save_preds_dir = os.path.join(SEGMENTATION_RESULTS_FOLDER, f'predictions')
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
    

    visualdir = os.path.join(SEGMENTATION_RESULTS_FOLDER, 'visualization', experiment_code)
    metricsdir = os.path.join(SEGMENTATION_RESULTS_FOLDER, 'testmetrics', experiment_code)
    os.makedirs(visualdir, exist_ok=True)
    os.makedirs(metricsdir, exist_ok=True)

    datapath = os.path.join(WORKING_FOLDER, 'data_analysis', 'datainfo.csv')
    data = pd.read_csv(datapath)
    test_ids = data[data['CenterID'] == args.leave_one_center_out]['PatientID'].tolist()
    stpaths_test = sorted([os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in test_ids])
    gtpaths_test = sorted([os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in test_ids])
    prpaths_test = sorted([os.path.join(save_preds_dir, f'{id}.nii.gz') for id in test_ids])
    fnames = [f'{os.path.basename(path).split(".")[0]}' for path in gtpaths_test]
    dsc_list = []


    for stpath, gtpath, prpath in zip(stpaths_test, gtpaths_test, prpaths_test):
        st, gt, pr = read_image_array(stpath), read_image_array(gtpath), read_image_array(prpath)
        fname = os.path.basename(gtpath).split(".")[0]
        dsc = calculate_patient_level_dice_score(gt, pr)
        dsc_list.append(dsc)
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        ax[0].imshow(st)
        ax[1].imshow(gt)
        ax[2].imshow(pr)
        ax[0].set_title('Image', fontsize=18)
        ax[1].set_title('GT', fontsize=18)
        ax[2].set_title('Pred', fontsize=18)
        # plt.show()
        figpath = os.path.join(visualdir, f'{fname}.png')
        fig.savefig(figpath, dpi=200, bbox_inches='tight')
        plt.close('all')
        print(f'{fname}: {dsc:2f}')
    
    
    data = pd.DataFrame(columns=['PatientID', 'DSC'])
    data['PatientID'] = fnames
    data['DSC'] = dsc_list
    metricspath = os.path.join(metricsdir, 'testmetrics.csv')
    data.to_csv(metricspath, index=False)


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
                        help='batchsize for sliding window inference (default=4)')
    parser.add_argument('--val-interval', type=int, default=2, metavar='val-interval',
                        help='epochs interval for which validation will be performed (default=2)')
    args = parser.parse_args()
    
    main(args)