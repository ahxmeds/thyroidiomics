#%%
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import sys 
import os 
import SimpleITK as sitk 
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER, DATA_FOLDER
# %%
def calculate_patient_level_dice_score(
    gtarray: np.ndarray,
    predarray: np.ndarray, 
) -> np.float64:
    """Function to return the Dice similarity coefficient (Dice score) between
    2 segmentation masks (containing 0s for background and 1s for lesions/tumors)

    Args:
        maskarray_1 (np.ndarray): numpy ndarray for the first mask
        maskarray_2 (np.ndarray): numpy ndarray for the second mask

    Returns:
        np.float64: Dice score
    """
    dice_score = 2.0*np.sum(predarray[gtarray == 1])/(np.sum(gtarray) + np.sum(predarray))
    return dice_score
def read_image_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T
#%%
network = 'unet1'
leave_one_center_out = 'I'
experiment_code = f"{network}_loco{leave_one_center_out}_trn64_val128_WDL"
preddir = os.path.join(RESULTS_FOLDER, 'predictions', experiment_code)
visualdir = os.path.join(RESULTS_FOLDER, 'visualization', experiment_code)
metricsdir = os.path.join(RESULTS_FOLDER, 'testmetrics', experiment_code)
os.makedirs(visualdir, exist_ok=True)
os.makedirs(metricsdir, exist_ok=True)
# find the best model for this experiment from the training/validation logs
# best model is the model with the best validation `Metric` (DSC)
datapath = '/home/jhubadmin/Projects/thyroid-segmentation/segmentation/datainfo_images_to_use.csv'
data = pd.read_csv(datapath)
data = data[data['Class'] != 0] # remove normal class 
test_ids = data[data['CenterID'] == leave_one_center_out]['PatientID'].tolist()
stpaths_test = sorted([os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in test_ids])
gtpaths_test = sorted([os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in test_ids])
prpaths_test = sorted([os.path.join(preddir, f'{id}.nii.gz') for id in test_ids])
fnames = [f'{os.path.basename(path).split(".")[0]}' for path in gtpaths_test]
dsc_list = []
# %%
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
# %%

data = pd.DataFrame(columns=['PatientID', 'DSC'])
data['PatientID'] = fnames
data['DSC'] = dsc_list
metricspath = os.path.join(metricsdir, 'testmetrics.csv')
data.to_csv(metricspath, index=False)
#%%
centers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
network = 'unet1'
for c in centers:
    experiment_code = f"{network}_loco{c}_trn64_val128_WDL"
    metricsdir = os.path.join(RESULTS_FOLDER, 'testmetrics', experiment_code)
    metricspath = os.path.join(metricsdir, 'testmetrics.csv')
    data = pd.read_csv(metricspath)
    print(f"{c} | Mean DSC = {data['DSC'].mean():.3f} +/- {data['DSC'].std():.3f}")

# %%
