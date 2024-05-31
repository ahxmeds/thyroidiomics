import os 

THYROIDIOMICS_FOLDER = '' # path to the directory containing `data` and `results` (this will be created by the pipeline) folders.
DATA_FOLDER = os.path.join(THYROIDIOMICS_FOLDER, 'data', 'nifti') # place your data in this location

SEGMENTATION_RESULTS_FOLDER = os.path.join(THYROIDIOMICS_FOLDER, 'segmentation_results')
CLASSIFICATION_RESULTS_FOLDER = os.path.join(THYROIDIOMICS_FOLDER, 'classification_results')
os.makedirs(SEGMENTATION_RESULTS_FOLDER, exist_ok=True)
os.makedirs(CLASSIFICATION_RESULTS_FOLDER, exist_ok=True)
WORKING_FOLDER = os.path.dirname(os.path.abspath(__file__))