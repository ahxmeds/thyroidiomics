# %%
# Importing the required libraries
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import argparse
import sys
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import (
    CLASSIFICATION_RESULTS_FOLDER, 
    SEGMENTATION_RESULTS_FOLDER, 
    DATA_FOLDER,
    WORKING_FOLDER
)
# %%
class RadiomicsFeatureExtractor:
    def __init__(self, params):
        """
        Initialize the RadiomicsFeatureExtractor with specified parameters.
        """
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    def extract_features(self, stpaths, gtpaths, classes, centers, output_excel_file):
        """
        Extract radiomics features from given paths for images and labels.
        """
        sample_names = []
        feature_data = []

        # Process each image-label pair
        for image_path, label_path in zip(stpaths, gtpaths):
            try:
                label_image = sitk.ReadImage(label_path)
                image = sitk.ReadImage(image_path)
                features = self.extractor.execute(image, label_image)

                sample_name = os.path.basename(image_path).split('.')[0]
                sample_names.append(sample_name)
                feature_data.append(features)
            except Exception as e:
                print(f"Error processing {image_path} or {label_path}: {e}")

        # Create and save the DataFrame
        # if feature_data:
        df = pd.DataFrame(feature_data)
        df.insert(0, 'PatientID', sample_names)
        df.insert(1, 'CenterID', centers)
        df.insert(2, 'Class', classes)
        # Ensure the output directory exists
        # os.makedirs(os.path.dirname(output_excel_file), exist_ok=True)
        df.to_excel(output_excel_file, index=False)
        print("Radiomics features saved to", output_excel_file)
        
#%%
class DataProcessor:
    def __init__(self, threshold=0.90):
        self.scaler = StandardScaler()
        self.threshold = threshold

    def scale_data(self, *datasets):
        self.scaler.fit(datasets[0])
        names = datasets[0].columns
        scaled_datasets = [pd.DataFrame(self.scaler.transform(dataset), columns=names) for dataset in datasets]
        return scaled_datasets

    def remove_highly_correlated_features(self, *datasets):
        correlation_matrix = datasets[0].corr(method='spearman').abs()
        mask = (correlation_matrix >= self.threshold) & (correlation_matrix < 1.0)
        features_to_remove = set()
        for feature in mask.columns:
            correlated_features = mask.index[mask[feature]].tolist()
            features_to_remove.update(set(correlated_features))
        filtered_datasets = [dataset.drop(columns=features_to_remove) for dataset in datasets]
        return filtered_datasets

    def process_data(self, X_train, X_test, X_External_1=None, X_External_2=None):
        if X_External_1 is not None and X_External_2 is not None:
            scaled_datasets = self.scale_data(X_train, X_test, X_External_1, X_External_2)
        elif X_External_1 is not None:
            scaled_datasets = self.scale_data(X_train, X_test, X_External_1)
        else:
            scaled_datasets = self.scale_data(X_train, X_test)
        
        filtered_datasets = self.remove_highly_correlated_features(*scaled_datasets)
        return filtered_datasets

class FeatureSelector:
    def __init__(self, method='rfe', estimator=None, n_features=10, direction='forward'):
        self.method = method.lower()
        self.n_features = n_features
        self.direction = direction
        self.estimator = estimator if estimator is not None else XGBClassifier()
        self.selector = None
        self.feature_scores_ = None

    def fit(self, X, y):
        if self.method == 'rfe':
            self.selector = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features
            )
        else:
            raise ValueError("Unsupported method")

        self.selector.fit(X, y)
        if hasattr(self.selector, 'estimator_'):
            self.feature_scores_ = self.selector.estimator_.feature_importances_ 
        elif hasattr(self.selector, 'scores_'):
            self.feature_scores_ = self.selector.scores_
        else:
            self.feature_scores_ = np.zeros(X.shape[1])

    def transform(self, X):
        if self.selector is None:
            raise RuntimeError("The selector has not been fitted yet.")
        
        return X.iloc[:, self.selector.get_support()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self, feature_names):
        if self.selector is None:
            raise RuntimeError("The selector has not been fitted yet.")
        
        return feature_names[self.selector.get_support()].tolist()

    def get_feature_scores(self):
        if self.feature_scores_ is None:
            raise RuntimeError("Feature scores are not available.")
        
        return self.feature_scores_
    
class MultiModelClassifier:
    def __init__(self):
        self.models = {'XGB': XGBClassifier(random_state=42)}
        
        self.param_grids = {'XGB': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_child_weight': [1, 5, 10]}}


    def train_model(self, model_name, X_train, y_train):
        if model_name not in self.models:
            raise ValueError("Invalid model name. Choose from: XGB")

        model = self.models[model_name]
        param_grid = self.param_grids[model_name]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train.values.ravel())

        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

        best_model = grid_search.best_estimator_

        return best_model
    

    def evaluate_model(self, best_model, X_test, y_test, plot_path=None):
        # Binarize the labels for ROC curve plotting
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        y_pred_proba_test = best_model.predict_proba(X_test)
        y_pred_test = best_model.predict(X_test)


        # Calculate ROC AUC for each class
        n_classes = y_test_binarized.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Weighted-average ROC curve and ROC area
        weights = np.sum(y_test_binarized, axis=0)
        weighted_mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            weighted_mean_tpr += weights[i] * np.interp(all_fpr, fpr[i], tpr[i])
        weighted_mean_tpr /= np.sum(weights)
        fpr["weighted"] = all_fpr
        tpr["weighted"] = weighted_mean_tpr
        roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        class_colors = cycle(['blue', 'red', 'green', 'purple'])

        for i, color in zip(range(n_classes), class_colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        # Plot ROC curve for micro-average
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=2,
                label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

        # Plot ROC curve for macro-average
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=2,
                label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

        # Plot ROC curve for weighted-average
        plt.plot(fpr["weighted"], tpr["weighted"], color='orange', linestyle='--', lw=2,
                label='Weighted-average ROC curve (area = {0:0.2f})'.format(roc_auc["weighted"]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for {}'.format(best_model.__class__.__name__))
        plt.legend(loc="lower right")

            # Save the ROC plot if plot_path is specified
        if plot_path:
            roc_plot_path = os.path.join(plot_path, f'{best_model.__class__.__name__}_ROC.png')
            plt.savefig(roc_plot_path)
            plt.close()  # Close the ROC plot

        # Calculate Precision-Recall for each class
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
            average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_proba_test[:, i])

        # Micro-average Precision-Recall curve and PR area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(),
                                                                        y_pred_proba_test.ravel())
        average_precision["micro"] = average_precision_score(y_test_binarized.ravel(), y_pred_proba_test.ravel(), average="micro")

        # Macro-average Precision-Recall curve and PR area
        precision["macro"], recall["macro"], _ = precision_recall_curve(y_test_binarized.ravel(),
                                                                        y_pred_proba_test.ravel())
        average_precision["macro"] = average_precision_score(y_test_binarized.ravel(), y_pred_proba_test.ravel(), average="macro")

        # Weighted-average Precision-Recall curve and PR area
        precision["weighted"], recall["weighted"], _ = precision_recall_curve(y_test_binarized.ravel(),
                                                                            y_pred_proba_test.ravel())
        average_precision["weighted"] = average_precision_score(y_test_binarized.ravel(), y_pred_proba_test.ravel(), average="weighted")

        # Plot Precision-Recall curves
        plt.figure(figsize=(8, 6))
        class_colors = cycle(['blue', 'red', 'green', 'purple'])
        average_colors = cycle(['cyan', 'magenta', 'yellow'])

        for i, color in zip(range(n_classes), class_colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label='PR curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

        # Plot Precision-Recall curves for micro, macro, and weighted averages
        for average, color in zip(['micro', 'macro', 'weighted'], average_colors):
            plt.plot(recall[average], precision[average], color=color, lw=2,
                    label='{0}-average Precision-Recall curve (area = {1:0.2f})'.format(average, average_precision[average]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for {}'.format(best_model.__class__.__name__))
        plt.legend(loc="lower left")

        # Save the plots if plot_path is specified
        if plot_path:
            pr_plot_path = os.path.join(plot_path, f'{best_model.__class__.__name__}_PRC.png')
            plt.savefig(pr_plot_path)
            plt.close()  # Close the PRC plot

        # Calculate confusion matrix
        conf_matrix_test = confusion_matrix(y_test, y_pred_test)
        print("Confusion Matrix (Test Set):\n", conf_matrix_test)

        # Evaluation Metrics on test
        accuracy_score_test = accuracy_score(y_test, y_pred_test)
        classification_report_test = classification_report(y_test, y_pred_test)
        print("Accuracy on Test Set: {:.2f}".format(accuracy_score_test))
        print("Classification Report:")
        print(classification_report_test)

        return accuracy_score_test, classification_report_test, conf_matrix_test

def group_data_by_centers(excel_file_path):
    # # Load the datasets
    df_features = pd.read_excel(excel_file_path, engine='openpyxl')
    centerids = np.unique(df_features['CenterID'].tolist())
    grouped_data = {}

    for center in centerids:
        df_center = df_features[df_features['CenterID'] == center]
        df_center.reset_index(inplace=True, drop=True)
        grouped_data[center] = df_center
   
    return grouped_data

def one_center_out_cross_validation(center_key, data_dict, data_dict_Predicted, results_path):
    selected_features_scores = {}  # Dictionary to save selected features with their scores

    test_set = data_dict[center_key]
    test_set_predicted = data_dict_Predicted[center_key]
    train_sets = [data_dict[key] for key in data_dict if key != center_key]
    train_set = pd.concat(train_sets, ignore_index=True)

    # Assuming the column name you want to start with is stored in a variable
    start_column_name = 'original_firstorder_10Percentile'
    # Find the index of the starting column
    start_index = train_set.columns.get_loc(start_column_name)
    X_train = train_set.iloc[:, start_index:]  
    y_train = train_set.iloc[:, 2:3]
    X_test = test_set.iloc[:, start_index:]
    y_test = test_set.iloc[:, 2:3]
    X_test_predicted = test_set_predicted.iloc[:, start_index:]
    y_test_predicted = test_set_predicted.iloc[:, 2:3]

    print(f"Center: {center_key}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"X_test_predicted shape: {X_test_predicted.shape}, y_test_predicted shape: {y_test_predicted.shape}")

    # Ensure the target variable is categorical
    y_train_modified = y_train.astype('category')
    y_test_modified = y_test.astype('category')
    y_test_predicted_modified = y_test_predicted.astype('category')

    processor = DataProcessor(threshold=0.95)
    scaled_and_filtered_data = processor.process_data(X_train, X_test, X_test_predicted)
    X_train_processed, X_test_processed, X_test_predicted_processed = scaled_and_filtered_data

    fs_rfe = FeatureSelector(method='rfe', estimator=XGBClassifier(), n_features=10)
    fs_rfe.fit(X_train_processed, y_train_modified)
    X_train_rfe = fs_rfe.transform(X_train_processed)
    X_test_rfe = fs_rfe.transform(X_test_processed)
    X_test_predicted_rfe = fs_rfe.transform(X_test_predicted_processed)

    selected_features_rfe = fs_rfe.get_selected_features(X_train_processed.columns)
    feature_scores_rfe = fs_rfe.get_feature_scores()

    # Save selected features and their scores in the dictionary
    selected_features_scores[center_key] = dict(zip(selected_features_rfe, feature_scores_rfe))

    directory_path = f'{results_path}/real_segs/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    mmc = MultiModelClassifier()
    best_models = {}
    for model_name in ['XGB']:
        best_models[model_name] = mmc.train_model(model_name, X_train_rfe, y_train_modified)

    with open(os.path.join(directory_path, 'classification_reports.txt'), 'w') as file:
        for model_name, model in best_models.items():
            print(f"Evaluating model: {model_name}")
            accuracy, class_report, conf_matrix = mmc.evaluate_model(model, X_test_rfe, y_test_modified, plot_path=directory_path)
            file.write(f"Model: {model_name}\n")
            file.write(f"Accuracy (without bootstrapping) on X_test: {accuracy:.2f}\n")
            file.write("Classification Report (without bootstrapping) on X_test:\n")
            file.write(class_report + "\n")
            file.write("Confusion Matrix (without bootstrapping) on X_test:\n")
            for row in conf_matrix:
                file.write(' '.join(str(x) for x in row) + "\n")
            file.write("-" * 80 + "\n")
    print("All model evaluations have been saved to 'classification_reports.txt'.")

    directory_path1 = f'{results_path}/predicted_segs/'
    if not os.path.exists(directory_path1):
        os.makedirs(directory_path1)

    with open(os.path.join(directory_path1, 'classification_reports.txt'), 'w') as file:
        for model_name, model in best_models.items():
            print(f"Evaluating model: {model_name}")
            accuracy, class_report, conf_matrix = mmc.evaluate_model(model, X_test_predicted_rfe, y_test_predicted_modified, plot_path=directory_path1)
            file.write(f"Model: {model_name}\n")
            file.write(f"Accuracy (without bootstrapping) on X_test_predicted: {accuracy:.2f}\n")
            file.write("Classification Report (without bootstrapping) on X_test_predicted:\n")
            file.write(class_report + "\n")
            file.write("Confusion Matrix (without bootstrapping) on X_test_predicted:\n")
            for row in conf_matrix:
                file.write(' '.join(str(x) for x in row) + "\n")
            file.write("-" * 80 + "\n")
    print("All model evaluations have been saved to 'classification_reports.txt'.")
    
    # Save the selected features and their scores to an Excel file
    excel_data = []
    for center_key, features_scores in selected_features_scores.items():
        for feature, score in features_scores.items():
            excel_data.append([center_key, feature, score])
    
    df_selected_features_scores = pd.DataFrame(excel_data, columns=['Center', 'Feature', 'Score'])
    features_scores_path = f'{results_path}/selected_features_scores.xlsx'
    df_selected_features_scores.to_excel(features_scores_path, index=False)
    
    print("Selected features and their scores have been saved to 'selected_features_scores.xlsx'.")
    return

#%%
def main(args):
    experiment_code = f'{args.network_name}_loco{args.leave_one_center_out}'
    feature_extraction_dir = os.path.join(CLASSIFICATION_RESULTS_FOLDER, 'feature_extraction')
    os.makedirs(feature_extraction_dir, exist_ok=True) 
    physiciansegs_feature_extraction_fpath = os.path.join(feature_extraction_dir, 'physician_segmentation_radiomics.xlsx')

    predictedsegs_feature_extraction_dir = os.path.join(CLASSIFICATION_RESULTS_FOLDER, 'feature_extraction', experiment_code)
    os.makedirs(predictedsegs_feature_extraction_dir, exist_ok=True)
    predictedsegs_feature_extraction_fpath = os.path.join(predictedsegs_feature_extraction_dir, 'predicted_segmentation_radiomics.xlsx')

    datapath = os.path.join(WORKING_FOLDER, 'data_analysis', 'datainfo.csv')
    data = pd.read_csv(datapath)
    trainvalid_df = data[data['CenterID'] != args.leave_one_center_out]
    test_df = data[data['CenterID'] == args.leave_one_center_out]
    trainvalid_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    trainvalid_ids, test_ids = trainvalid_df['PatientID'].tolist(), test_df['PatientID'].tolist()
    trainvalid_classes, test_classes = trainvalid_df['Class'].tolist(), test_df['Class'].tolist()
    trainvalid_centers, test_centers = trainvalid_df['CenterID'].tolist(), test_df['CenterID'].tolist()

    stpaths_trainvalid = [os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in trainvalid_ids]
    gtpaths_trainvalid = [os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in trainvalid_ids]

    stpaths_test = [os.path.join(DATA_FOLDER, 'images', f'{id}.nii.gz') for id in test_ids]
    gtpaths_test = [os.path.join(DATA_FOLDER, 'labels', f'{id}.nii.gz') for id in test_ids]

    predicted_labels_dir = os.path.join(SEGMENTATION_RESULTS_FOLDER, 'predictions', experiment_code)
    prpaths_test = [os.path.join(predicted_labels_dir, f'{id}.nii.gz') for id in test_ids]

    # Define the feature extractor parameters
    params = {
        "binWidth": 0.1,                 # 0.1
        "resampledPixelSpacing": (1, 1), # Resampling
        "interpolator": sitk.sitkBSpline,# Correct the interpolator setting
        "force2D": True,                 # Force the extractor to treat the images as 2D
        "normalize": True,               # Normalize the image before calculation
    }

    # Create an instance of the extractor
    feature_extractor = RadiomicsFeatureExtractor(params)

    # # Execute feature extraction
    # # physician's features should be extracted only once
    if os.path.exists(physiciansegs_feature_extraction_fpath) == False: 
        feature_extractor.extract_features(
            stpaths_trainvalid+stpaths_test, 
            gtpaths_trainvalid+gtpaths_test, 
            trainvalid_classes+test_classes, 
            trainvalid_centers+test_centers, 
            physiciansegs_feature_extraction_fpath
        )
    else:
        pass

    feature_extractor.extract_features(
        stpaths_test, 
        prpaths_test, 
        test_classes, 
        test_centers, 
        predictedsegs_feature_extraction_fpath
    )

    data_dict = group_data_by_centers(physiciansegs_feature_extraction_fpath) 
    data_dict_predicted = group_data_by_centers(predictedsegs_feature_extraction_fpath) 

    save_results_dir = os.path.join(CLASSIFICATION_RESULTS_FOLDER, 'predictions_and_metrics')
    os.makedirs(save_results_dir, exist_ok=True)
    save_results_dir = os.path.join(save_results_dir, experiment_code)
    os.makedirs(save_results_dir, exist_ok=True)

    one_center_out_cross_validation(args.leave_one_center_out, data_dict, data_dict_predicted, save_results_dir)
# %%
if __name__ == "__main__": 
    # create datasplit files for train and test images
    # follow all the instructions for dataset directory creation and images/labels file names as given in: LINK
    parser = argparse.ArgumentParser(description='Thyroid pathology classification in scintigraphy images')
    parser.add_argument('--network-name', type=str, default='unet', metavar='netname',
                        help='network name for training (default: unet)') 
    parser.add_argument('--leave-one-center-out', type=str, default='A', metavar='center',
                        help='leave a center out for testing (default: A)')
    args = parser.parse_args()
    
    main(args)
# %%
