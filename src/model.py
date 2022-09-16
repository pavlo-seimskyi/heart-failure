import os
import pickle
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict

from src.evaluate import generate_cv_splits, train_valid_split
from src.metrics import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


# TODO define a list of features and be able to drop some of them
class BinaryClassifier(object):
    def __init__(self, model, preprocessor):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = 0.5

    def fit(self, x, y, param_grid, metrics, **kwargs):
        grid_model = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=10,
            scoring=metrics,
            refit=metrics[0],
            **kwargs
        )
        grid_model.fit(x, y)
        self.model = grid_model.best_estimator_
        # TODO save the model

    def predict(self, x):
        """Predict with a custom threshold."""
        y_pred_proba = self.model.predict_proba(x)[:, 1]
        return (y_pred_proba >= self.threshold).astype(float)

    def preprocess(self, x_train, x_valid, y_train, y_valid):
        """Ensures preprocessing without data leakage."""
        x_train = self.preprocessor.encode_cat_feats(x_train)
        x_valid = self.preprocessor.encode_cat_feats(x_valid)
        # other preprocessing that involves fit_transform and transform
        return x_train, x_valid, y_train, y_valid

    def cross_val_predict(self, x, y, cv_folds=10):
        y_preds = None
        splits = generate_cv_splits(x.shape[0], cv_folds)
        for valid_start, valid_end in splits:
            x_train, x_valid, y_train, y_valid = train_valid_split(x, y, valid_start, valid_end)
            x_train, x_valid, y_train, y_valid = self.preprocess(x_train, x_valid, y_train, y_valid)
            model = deepcopy(self.model)  # the model with the current params
            model.fit(x_train, y_train)
            y_pred = model.predict_proba(x_valid)[:, 1]
            y_preds = y_pred if y_preds is None else np.append(y_preds, y_pred, axis=0)
        return y_preds

    def save_model(self, model_name, folder):
        """Save the model to disk."""
        path = os.path.join(folder, model_name)
        os.makedirs(path, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_name, folder):
        """Load the model from disk."""
        path = os.path.join(folder, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file/directory found at path: {path}")
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    # TODO write own function to evaluate: includes preprocessing
    # TODO Plus, stratified sampling in cross-validation?
    def evaluate(self, x, y, cv_folds=10):
        """
        Evaluate the current model's performance on all available data.
        :param x: all features
        :param y: all labels
        :param cv_folds: how many validation folds to run
        :return:
        """
        # TODO Load the model and only evaluate, not train
        y_pred_proba = self.cross_val_predict(x, y, cv_folds=cv_folds)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(float)
        print(classification_report(y, y_pred))
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        plot_confusion_matrix(y, y_pred, ax=axs[0])
        plot_precision_recall_curve(
            y, y_pred_proba, threshold=self.threshold, ax=axs[1]
        )
        plot_roc_curve(y, y_pred_proba, threshold=self.threshold, ax=axs[2])

    def select_threshold_based_on_recall(self, x, y, min_recall, cv=10):
        """
        Modify the cutoff point for binary prediction based on desired recall.
        Runs a full cross-validation cycle to get predictions.

        :param x: Input features (full dataset)
        :param y: Target variable (full dataset)
        :param min_recall: Desired minimal recall
        :param cv: Cross-validation folds
        :return: None
        """
        y_pred_proba = cross_val_predict(
            self.model, x, y, cv=cv, method="predict_proba"
        )[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        thresholds = np.append(thresholds, 1)
        self.threshold = thresholds[recall >= min_recall].max()
