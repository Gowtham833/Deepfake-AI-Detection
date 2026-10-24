import pytest
import numpy as np
from sklearn.metrics import f1_score

def test_default_threshold():
    # Usually 0.5
    threshold = 0.5
    assert threshold == 0.5

def test_validation_threshold_selection():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_probs = np.array([0.1, 0.4, 0.6, 0.3, 0.9, 0.2])
    
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_thresh = 0.5
    best_f1 = 0
    
    for t in thresholds:
        t_val = round(float(t), 2)
        preds = (y_probs >= t_val).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t_val
            
    # t=0.3 gives preds: [0, 1, 1, 1, 1, 0] -> TP=3, FP=1, FN=0 -> f1 = 3 / 3.5 = 0.857
    # t=0.4 gives preds: [0, 1, 1, 0, 1, 0] -> TP=3, FP=0, FN=0 -> f1 = 1.0
    assert np.isclose(best_thresh, 0.4)
