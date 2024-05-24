#%%
#%%
import os 
import SimpleITK as sitk 
import pydicom 
import numpy as np
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
from skimage.transform import resize
from skimage.morphology import remove_small_holes
import time
# %%
def read_nifti_image(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T

def read_dicom_image(path):
    dc = pydicom.dcmread(path)
    return dc.pixel_array, dc

#%%
dicomdir = '/data/blobfuse/default/thyroid-segmentation-results/data/dicom/Data_anonymous'
patientdirs_all = os.listdir(dicomdir)
patientdirs = []
for item in patientdirs_all:
    if item.startswith('E'):
        patientdirs.append(item)
stpaths = [os.path.join(dicomdir, d, f'{d}.dcm') for d in patientdirs]
gtpaths = [os.path.join(dicomdir, d, f'Untitled.nii.gz') for d in patientdirs]

savedir = '/data/blobfuse/default/thyroid-segmentation-results/data/'
niftisavedir = os.path.join(savedir, 'nifti')
vizsavedir = os.path.join(savedir, 'visualization')
os.makedirs(niftisavedir, exist_ok=True)
os.makedirs(vizsavedir, exist_ok=True)
#%%
def preprocess_one_image(stpath, gtpath):
    # read dicom, nifti images into numpy arrays 
    st, dc = read_dicom_image(stpath) # ST
    gt = read_nifti_image(gtpath) # GT
    # read spacing from nifti data
    spacing = np.array(sitk.ReadImage(gtpath).GetSpacing()[:-1])
    
    # find the number of dimensions in the ST 
    # it can be 2D (128,128) or 3D (2, 128, 128) or (2, 256,256)
    # goal is to get a final 2D image
    # select the channel indexed 0 when the image is 3D
    if st.ndim == 2:
        st_final = st 
    elif st.ndim == 3:
        st_final = st[0]
    else:
        pass
    
    # find the final GT array for the two cases (as above)
    # properly rotate/transpose to get GT in the correct orientation
    if gt.shape[2] == 1:
        gt_final = gt[:,:,0].T
    elif gt.shape[2] == 2: 
        gt_final = np.rot90(gt[:,:,1], 3)
    
    # resize the images to 128x128 if they are 256x256
    # update the spacing variable for such images
    if gt_final.shape[0] == 256:
        #
        print('inside 256 loop')
        st_final = resize(st_final, (128,128), order=0, anti_aliasing=True)
        gt_final = resize(gt_final, (128,128), order=0, anti_aliasing=False, preserve_range=True)
        print(f'Old spacing: {spacing}')
        spacing = 2*spacing
        print(f'New spacing: {spacing}')

    # fill holes, because that's Maziar's favourite activity
    gt_final = remove_small_holes(gt_final, area_threshold=10, connectivity=2).astype(int)

    # convert the numpy arrays to sitk nifti image class
    # note: transposing before converting is important
    stimg = sitk.GetImageFromArray(st_final.T)
    gtimg = sitk.GetImageFromArray(gt_final.T)

    # set the values for spacing
    stimg.SetSpacing(spacing)
    gtimg.SetSpacing(spacing)

    # define filenames for final ST and GT nifti images for saving purpose
    # save them as .nii.gz 
    # .nii.gz is the compressed version, has a little overhead when opening them
    # but saves disk space
    # since the images are already small, it is okay to save them as compressed files
    fname = os.path.basename(stpath).split('.')[0]
    stsavepath = os.path.join(niftisavedir, 'images', f'{fname}.nii.gz')
    gtsavepath = os.path.join(niftisavedir, 'labels', f'{fname}.nii.gz')
    sitk.WriteImage(stimg, stsavepath)
    sitk.WriteImage(gtimg, gtsavepath)

    # plot and save ST and GT side-by-side for visualization
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(st_final)
    # ax[1].imshow(gt_final)
    # vizsavepath = os.path.join(vizsavedir, f'{fname}.png')
    # fig.savefig(vizsavepath, dpi=200, bbox_inches='tight')
    plt.close('all')
    print(f'Done with {fname}')
    
    
# %%
# parallelize the preprocessing on n_jobs CPU cores using joblib module
start = time.time()
Parallel(n_jobs=8)(delayed(preprocess_one_image)(stpath, gtpath) for stpath, gtpath in zip(stpaths, gtpaths))
print(f'Time taken: {(time.time() - start)/60} mins')
# %%
