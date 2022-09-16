import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict

from src.metrics import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.utils import get_base_path


class BinaryClassifierModel(object):
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
            **kwargs,
        )
        grid_model.fit(x, y)
        self.model = grid_model.best_estimator_

    def predict(self, x):
        """Predict with a custom threshold."""
        y_pred_proba = self.model.predict_proba(x)[:, 1]
        return (y_pred_proba >= self.threshold).astype(float)

    def save_model(self, folder, model_name):
        """Save the model to disk."""
        folder_path = os.path.join(get_base_path(), folder)
        file_path = os.path.join(folder_path, model_name)
        os.makedirs(folder_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, folder, model_name):
        """Load the model from disk."""
        path = os.path.join(get_base_path(), folder, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file/directory found at path: {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess(self, x):
        return self.preprocessor.preprocess(x)

    def evaluate(self, x, y, cv=10):
        y_pred_proba = cross_val_predict(
            self.model, x, y, cv=cv, method="predict_proba"
        )[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(float)
        print(classification_report(y, y_pred))
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
        plt.tight_layout(h_pad=1, w_pad=9)
        plot_confusion_matrix(y, y_pred, ax=axs[0])
        plot_precision_recall_curve(
            y, y_pred_proba, threshold=self.threshold, ax=axs[1]
        )
        plot_roc_curve(y, y_pred_proba, threshold=self.threshold, ax=axs[2])
        plot_feature_importance(self.model, x.columns.tolist(), ax=axs[3])

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
