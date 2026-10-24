import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

# REAL = 0, DEEPFAKE = 1
y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
y_pred_probs = np.array([0.1, 0.9, 0.8, 0.4, 0.6, 0.2, 0.7, 0.9])
y_pred_labels = (y_pred_probs >= 0.5).astype(int)

def test_accuracy():
    acc = accuracy_score(y_true, y_pred_labels)
    assert acc == 0.875 # 7 out of 8 correct (index 6 is FP)

def test_precision():
    prec = precision_score(y_true, y_pred_labels)
    assert prec == 0.8 # 4 TP, 1 FP

def test_recall():
    rec = recall_score(y_true, y_pred_labels)
    assert rec == 1.0 # 4 TP, 0 FN

def test_f1():
    f1 = f1_score(y_true, y_pred_labels)
    assert np.isclose(f1, 0.8888, atol=1e-3)

def test_roc_auc():
    auc = roc_auc_score(y_true, y_pred_probs)
    assert auc > 0.8

def test_pr_auc():
    pr_auc = average_precision_score(y_true, y_pred_probs)
    assert pr_auc > 0.8

def test_confusion_matrix():
    cm = confusion_matrix(y_true, y_pred_labels)
    assert cm[0, 0] == 3 # TN
    assert cm[0, 1] == 1 # FP
    assert cm[1, 0] == 0 # FN
    assert cm[1, 1] == 4 # TP
