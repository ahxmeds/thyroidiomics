# Thyroidiomics: An Automated Pipeline for Segmentation and Classification of Thyroid Pathologies from Scintigraphy Images


## Introduction
This codebase is related to our submission to EUVIP 2024:<br>
> Anonymous authors, _Thyroidiomics: An Automated Pipeline for Segmentation and Classification of Thyroid Pathologies from Scintigraphy Images_.

<p align="center">
<img src="./assets/flowchart.png" alt="Figure" height="285" />
</p>
<p align="center", style="font-size:8px">
    Figure 1: <i>Thyroidiomics</i>: the proposed two-step pipeline to classify thyroid pathologies into three classes, namely, MNG, TH and DG. Scenario 1 represents the pipeline dependent on physician's delineated ROIs as input to the classifier, while scenario 2 represents the fully automated pipeline operating on segmentation predicted by ResUNet.
</p>

<p align="justify">
The objective of this study was to develop an automated pipeline that enhances thyroid disease classification using thyroid scintigraphy images, aiming to decrease assessment time and increase diagnostic accuracy. Anterior thyroid scintigraphy images from 2,643 patients were collected and categorized into diffuse goiter (DG), multinodal goiter (MNG), and thyroiditis (TH) based on clinical reports, and then segmented by an expert. A Residual UNet (ResUNet) model was trained to perform auto-segmentation. Radiomics features were extracted from both physician's (scenario 1) and ResUNet segmentations (scenario 2), followed by omitting highly correlated features using Spearman's correlation, and feature selection using Recursive Feature Elimination (RFE) with eXtreme Gradient Boosting (XGBoost) as the core. All models were trained under leave-one-center-out cross-validation (LOCOCV) scheme, where nine instances of algorithms was iteratively trained and validated on data from eight centers and tested on the ninth for both scenarios separately. Segmentation performance was assessed using the Dice similarity coefficient (DSC), while classification performance was assessed using metrics such as precision, recall, F1-score, accuracy, area under the Receiver Operating Characteristic (ROC AUC), and area under the precision-recall curve (PRC AUC). ResUNet obtained DSC values of 0.84±0.03, 0.71±0.06, and 0.86±0.02 for MNG, TH, and DG, respectively. Classification in scenario 1 achieved an accuracy of 0.76±0.04 and a ROC AUC of 0.92±0.02 while in scenario 2, classification yielded an accuracy of 0.74±0.05 and a ROC AUC of 0.90±0.02. The automated pipeline demonstrated comparable performance to physician segmentations on several classification metrics across different classes, effectively reducing assessment time while maintaining high diagnostic accuracy.
</p>

### Segmentation 
<p align="center">
<img src="./assets/segmentation.png" alt="Figure" height="300" />
</p>
<p align="center", style="font-size:8px">
    Figure 2: (a) Distribution of center-level mean DSC over 9 centers for the classes, MNG, TH and DG. (b)-(d), (e)-(g), and (h)-(j) show some representative images from each class with the ground truth (red) and ResUNet predicted (yellow) segmentation of thyroid. The DSC between ground truth and predicted masks is shown in the bottom-right of each figure.
</p>

### Classification
<p align="center">
<img src="./assets/classification.png" alt="Figure" height="500" />
</p>
<p align="center", style="font-size:8px">
    Figure 3: Various class-wise and averaged metrics for classification were used to evaluate model performance in two scenarios: features extracted from the physician's delineated ROIs and those from ResUNet predicted ROIs. The boxplots show the distribution of metrics over the nine centers as test sets for the three thyroid pathology classes, MNG, TH and DG. The black horizontal lines denote the median and white circle denote the mean of distribution.
</p>



## How to get started?
Follow the intructions given below to set up the necessary conda environment, install packages, preprocess dataset in the correct format so it can be accepted as inputs by the code, train model and perform anomaly detection on test set using the trained models. 

- **Clone the repository, create conda environment and install necessary packages:** The first step is to clone this GitHub codebase in your local machine, <!--create a conda environment, and install all the necessary packages. This code base was developed primarily using python=3.8.10, PyTorch=1.11.0, monai=1.3.0, with CUDA 11.3 on an Ubuntu 20.04 virtual machine, so the codebase has been tested only with these configurations. We hope that it will run in other suitable combinations of different versions of python, PyTorch, monai, and CUDA, but we cannot guarantee that. Proceed with caution! -->
 <!--   
    ```
    git clone 'https://github.com/igcondapet/IgCONDA-PET.git'
    cd IgCONDA-PET
    conda env create --file requirements.yaml
    ```
    The last step above creates a conda environment named `igcondapet`. Make sure you have conda installed. Next, activate the conda environment
    ```
    conda activate igcondapet
    ``'-->
- **Preprocess AutoPET and HECKTOR datasets:** 

<!--Go to [config.py](config.py) and set path to data folders for AutoPET and HECKTOR datasets and the path where you want the preprocessed data to be stored. 
    ```
    path_to_autopet_data_dir = '' # path to AutoPET data
    path_to_hecktor_data_dir = '' # path to hecktor data 
    path_to_preprocessed_data_dir = '' # path to directory where you want to store your preprocessed data
    ```
    The directory structure within `path_to_autopet_data_dir` and `path_to_hecktor_data_dir` should be as shown below. The folders `images` and `labels` must contain all the 3D PET and ground truth images for each datasets in `.nii.gz` format. Notice, the PET image filenames (excluding `.nii.gz` extension) end with `_0001` for AutoPET data, and with `__PT` for HECKTOR data. Ensure that your files are renamed in this format before proceeding. Information about the images included in the training, validation and test phases in this work are given in [data_split/metadata3D.csv](data_split/metadata3D.csv). Your filenames for PET and ground truth images should exactly match the filenames in the columns `PTPATH` and `GTPATH` in this file. 

        └───path_to_autopet_data_dir/
            ├── images
            │   ├── Patient0001_0001.nii.gz
            │   ├── Patient0002_0001.nii.gz
            │   ├── ...
            ├── labels
            │   ├── Patient0001.nii.gz
            │   ├── Patient0002.nii.gz 
            │   ├── ...

        └───path_to_hecktor_data_dir/
            ├── images
            │   ├── Patient0261__PT.nii.gz
            │   ├── Patient0262__PT.nii.gz
            │   ├── ...
            ├── labels
            │   ├── Patient0261.nii.gz
            │   ├── Patient0262.nii.gz 
            │   ├── ...

    This step centrally crops and downsamples the images (from both the datasets) to `64 x 64 x 96` and then saves the individual axial slices (96 slices per image) to your local device. The downsampled 3D images and 2D images are stored under `path_to_preprocessed_data_dir/preprocessed3D` and  `path_to_preprocessed_data_dir/preprocessed2D`, respectively. If your data is stored exactly as shown in the schematic above, run the data preprocessing step using [preprocess_data/preprocess_data.py](preprocess_data/preprocess_data.py)
    ```
    cd preprocess_data
    python preprocess_data.py
    ```
-->
- **Run training:** 
<!--The file [igcondapet/trainddp.py](igcondapet/trainddp.py) runs training on the 2D dataset via PyTorch's `DistributedDataParallel`. To run training, do the following (an example bash script is given in [igcondapet/trainddp.sh](igcondapet/trainddp.sh)):
    ```
    cd igcondapet
    torchrun --standalone --nproc_per_node=1 trainddp.py --experiment='exp0' --attn-layer1=False --attn-layer2=True --attn-layer3=True --epochs=400 --batch-size=32 --num-workers=4 --cache-rate=1.0 --val-interval=10
    ```
    Use a unique `--experiment` flag everytime you run a new training. Set `--nproc_per_node` as the number of GPU nodes available for parallel training. The data is cached using MONAI's `CacheDataset`, so if you are running out of memory, consider lowering the value of `cache_rate`. During training, the training and validation losses are saved under `./results/logs/trainlog_gpu{rank}.csv` and `./results/logs/validlog_gpu{rank}.csv` where `{rank}` is the GPU rank and updated every epoch. The checkpoints are saved every `val_interval` epochs under `./results/models/checkpoint_ep{epoch_number}.pth`.
-->
- **Run evaluation on test set:** 
<!--After the training is finished (for a given `--experiment`), [igcondapet/inference.py](igcondapet/inference.py) can be used to run evaluation on the unhealthy 2D test dataset and save the results. To run test evaluation, do the following (an example bash script is given in [igcondapet/inference.py](igcondapet/inference.py)):
    ```
    cd igcondapet
    python inference.py --experiment='exp0' --attn-layer1=False --attn-layer2=True --attn-layer3=True --guidance-scale=3.0 --noise-level=400 --num-workers=4 --cache-rate=1.0 --val-interval=10
    ```
    [igcondapet/inference.py](igcondapet/inference.py) uses the model with the lowest loss on the validation set for test evaluation. CAUTION: set `--val-interval` to the same value that was used during training.
-->

# References

<a id="1">[1]</a> 


<a id="2">[2]</a> 


<a id="3">[3]</a> 

<a id="4">[4]</a> 
