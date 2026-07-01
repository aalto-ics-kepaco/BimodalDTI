import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import matplotlib

def calc_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred, event_observed=np.ones(len(y_true)))
    log_cosh = np.mean(np.log(np.cosh(y_pred - y_true)))
    return {
        'mse': mse,
        'rmse': rmse,
        'spearman correlation': spearman,
        'pearson correlation': pearson,
        'concordance index': ci,
        'log cosh': log_cosh
    }

def plot_predictions(y_true: list[float], y_pred: list[float], save_file: str | None = None):
    plt.figure()
    plt.hexbin(y_true, y_pred, gridsize=30, cmap='Blues', norm=matplotlib.colors.LogNorm())
    plt.xlabel('true pChEMBL value')
    plt.ylabel('predicted pChEMBL value')
    plt.title('pChEMBL value predictions')
    plt.colorbar()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)

def show_metrics(y_true: list[float], y_pred: list[float], plot=True):
    if plot:
        plot_predictions(y_true, y_pred, './predictions.png')
    metrics = calc_metrics(y_true, y_pred)
    lines = []
    for key, value in metrics.items():
        line = f'{key}: {value}'
        lines.append(line)
        print(line)
    return '\n'.join(lines)
