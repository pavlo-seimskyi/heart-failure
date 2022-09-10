import pandas as pd
from sklearn.metrics import RocCurveDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_feature_importance(model, x_cols, ascending=False):
    return pd.DataFrame(
        {"cols": x_cols, "imp": model.feature_importances_}
    ).sort_values("imp", ascending=ascending)


def plot_confusion_matrix(y_test, y_pred):
    sns.heatmap(
        data=confusion_matrix(y_test, y_pred),
        annot=True,
        cmap="Blues",
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix")
    plt.show()


def plot_roc_curve(y_test, y_pred):
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.show()
