# Thyroidiomics: An Automated Pipeline for Segmentation and Classification of Thyroid Pathologies from Scintigraphy Images


## Introduction
This codebase is related to our submission to EUVIP 2024:<br>
> Anonymous authors, _Thyroidiomics: An Automated Pipeline for Segmentation and Classification of Thyroid Pathologies from Scintigraphy Images_.

<p align="center">
<img src="./assets/flowchart.png" alt="Figure" height="285" />
</p>
<p align="center", style="font-size:10px">
    <i>Thyroidiomics</i>: the proposed two-step pipeline to classify thyroid pathologies into three classes, namely, MNG, TH and DG. Scenario 1 represents the pipeline dependent on physician's delineated ROIs as input to the classifier, while scenario 2 represents the fully automated pipeline operating on segmentation predicted by ResUNet.
</p>

<p align="justify">
The objective of this study was to develop an automated pipeline that enhances thyroid disease classification using thyroid scintigraphy images, aiming to decrease assessment time and increase diagnostic accuracy. Anterior thyroid scintigraphy images from 2,643 patients were collected and categorized into diffuse goiter (DG), multinodal goiter (MNG), and thyroiditis (TH) based on clinical reports, and then segmented by an expert. A Residual UNet (ResUNet) model was trained to perform auto-segmentation. Radiomics features were extracted from both physician's (scenario 1) and ResUNet segmentations (scenario 2), followed by omitting highly correlated features using Spearman's correlation, and feature selection using Recursive Feature Elimination (RFE) with eXtreme Gradient Boosting (XGBoost) as the core. All models were trained under leave-one-center-out cross-validation (LOCOCV) scheme, where nine instances of algorithms was iteratively trained and validated on data from eight centers and tested on the ninth for both scenarios separately. Segmentation performance was assessed using the Dice similarity coefficient (DSC), while classification performance was assessed using metrics such as precision, recall, F1-score, accuracy, area under the Receiver Operating Characteristic (ROC AUC), and area under the precision-recall curve (PRC AUC). ResUNet obtained DSC values of 0.84±0.03, 0.71±0.06, and 0.86±0.02 for MNG, TH, and DG, respectively. Classification in scenario 1 achieved an accuracy of 0.76±0.04 and a ROC AUC of 0.92±0.02 while in scenario 2, classification yielded an accuracy of 0.74±0.05 and a ROC AUC of 0.90±0.02. The automated pipeline demonstrated comparable performance to physician segmentations on several classification metrics across different classes, effectively reducing assessment time while maintaining high diagnostic accuracy.
</p>

### Segmentation 
<p align="center">
<img src="./assets/segmentation.png" alt="Figure" height="300" />
</p>
<p align="center", style="font-size:10px">
    (a) Distribution of center-level mean DSC over 9 centers for the classes, MNG, TH and DG. (b)-(d), (e)-(g), and (h)-(j) show some representative images from each class with the ground truth (red) and ResUNet predicted (yellow) segmentation of thyroid. The DSC between ground truth and predicted masks is shown in the bottom-right of each figure.
</p>

### Classification
<p align="center">
<img src="./assets/classification.png" alt="Figure" height="400" />
</p>
<p align="center", style="font-size:10px">
    Various class-wise and averaged metrics for classification were used to evaluate model performance in two scenarios: features extracted from the physician's delineated ROIs and those from ResUNet predicted ROIs. The boxplots show the distribution of metrics over the nine centers as test sets for the three thyroid pathology classes, MNG, TH and DG. The black horizontal lines denote the median and white circle denote the mean of distribution.
</p>