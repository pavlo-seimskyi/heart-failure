import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict

from src.metrics import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


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
            **kwargs
        )
        grid_model.fit(x, y)
        self.model = grid_model.best_estimator_

    def predict(self, x):
        """Predict with a custom threshold."""
        y_pred_proba = self.model.predict_proba(x)[:, 1]
        return (y_pred_proba >= self.threshold).astype(float)

    def preprocess(self, x):
        return self.preprocessor.preprocess(x)

    def evaluate(self, x, y, cv=10):
        y_pred_proba = cross_val_predict(
            self.model, x, y, cv=cv, method="predict_proba"
        )[:, 1]
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
