from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

def return_replicates(x, y):
    "Stratified sampling followed by 2 sets of train and test datasets"
    x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5, stratify=y)
    
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, stratify=y1)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.25, stratify=y2)
    
    return  x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test 
                                                       

def expected_calibration_error(y_true, y_pred, num_bins=15):
    "y_preds should contain list of probabilities. More bins reduce the bias, but increase the variance "
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0], accuracy_score(y_true, pred_y)

def ause(pred_probs, pred_variances, true_labels):
    num_pred = pred_probs.shape[0]
    error_res = (true_labels - pred_probs)**2
    idx_res = np.argsort(error_res)
    idx_var = np.argsort(pred_variances)
    fractions = list(np.arange(start=0.,stop=1.,step=0.01))
    var_rmses, error_rmses = [], []
    for step, fraction in enumerate(fractions):
        if int((1.-fraction)*num_pred) == 0:
            var_rmses.append(0)
            error_rmses.append(0)
            continue
        var_rmse = np.sqrt(np.mean(error_res[idx_var[0:int((1.-fraction)*num_pred)]]))
        var_rmses.append(var_rmse)
        
        error_rmse = np.sqrt(np.mean(error_res[idx_res[0:int((1.-fraction)*num_pred)]]))
        error_rmses.append(error_rmse)
    error_rmses_normalized = error_rmses/error_rmses[0]
    var_rmses_normalized = var_rmses/var_rmses[0]
    sparsification_errors = var_rmses_normalized - error_rmses_normalized
    ause = np.trapz(y=sparsification_errors, x=fractions)
    return ause, sparsification_errors, error_rmses_normalized, var_rmses_normalized, fractions