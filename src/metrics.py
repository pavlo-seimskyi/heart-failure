import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, precision_score, recall_score,
                             roc_curve)


def plot_feature_importance(model, x_cols, ax):
    importances = get_feature_importance(model, x_cols)
    sns.barplot(data=importances, y="cols", x="imp", ax=ax)
    plt.ylabel("")
    plt.xlabel("relative importance")
    plt.title("Feature importance")
    plt.show()


def get_feature_importance(model, x_cols, ascending=False):
    return pd.DataFrame(
        {"cols": x_cols, "imp": model.feature_importances_}
    ).sort_values("imp", ascending=ascending)


def plot_precision_recall_curve(y_test, y_pred_proba, threshold, ax):
    """Plot the curve with a marker pointing to the current cutoff threshold.

    :param y_test: true target variable
    :param y_pred_proba: prediction probabilities for the positive class (0.0 to 1.0)
    :param threshold: cutoff point to determine the final prediction
    :param ax: matplotlib.plt axis
    :return: None
    """
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
    ax.set_title("Precision-recall curve")
    y_pred = (y_pred_proba >= threshold).astype(float)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ax.plot(
        recall,
        precision,
        marker="x",
        color="r",
        markersize=15,
        label="chosen threshold",
    )
    ax.legend()


def plot_roc_curve(y_test, y_pred_proba, threshold, ax):
    """
    Plot the curve with a marker pointing to the current cutoff threshold.

    :param y_test: true target variable
    :param y_pred_proba: prediction probabilities for the positive class (0.0 to 1.0)
    :param threshold: cutoff point to determine the final prediction
    :param ax: matplotlib.plt axis
    :return: None
    """
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
    ax.set_title("ROC curve")
    fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred_proba)
    nearest_thres_idx = abs((thresholds - threshold)).argmin()
    ax.plot(
        fp_rate[nearest_thres_idx],
        tp_rate[nearest_thres_idx],
        marker="x",
        color="r",
        markersize=15,
        label="chosen threshold",
    )
    ax.legend()


def plot_confusion_matrix(y_test, y_pred, ax):
    """
    Plot a confusion matrix on axis.

    :param y_test: true target variable
    :param y_pred: predictions
    :param ax: matplotlib.plt axis
    :return:
    """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix")
