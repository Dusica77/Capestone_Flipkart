"""Shared helpers for the capstone. Importing this file runs nothing: it only defines functions."""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# column positions inside the array returned by load_data()
# 0..6 = X1..X7, 7 = outcome (churned), 8 = value (order_value_inr_hundreds)
OUTCOME, VALUE = 7, 8


def path(*parts):
    return os.path.join(ROOT, *parts)


def load_data():
    with open(path("data", "records.csv")) as f:
        reader = csv.reader(f)
        next(reader)
        rows = [r for r in reader]
    return np.array([[float(v) for v in r[1:]] for r in rows])


# ---------------------------------------------------------------- output helpers
def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def write_results(name, items):
    os.makedirs(path("outputs"), exist_ok=True)
    with open(path("outputs", name), "w") as f:
        for k, v in items:
            f.write(f"{k} = {fmt(v)}\n")


def write_csv(name, header, rows):
    os.makedirs(path("outputs"), exist_ok=True)
    with open(path("outputs", name), "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


# ---------------------------------------------------------------- Part 2 building blocks
def stratified_split(data):
    """70/30 split per class in file order (integer arithmetic), both sets keep file order."""
    y = data[:, OUTCOME]
    tr, te = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        k = (7 * len(idx)) // 10
        tr += list(idx[:k])
        te += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


def prepare(data):
    """Split + standardise with TRAINING statistics only."""
    tr, te = stratified_split(data)
    Xtr_raw, Xte_raw = data[tr, :7], data[te, :7]
    mu = Xtr_raw.mean(axis=0)
    sd = Xtr_raw.std(axis=0, ddof=1)
    return {
        "tr": tr, "te": te,
        "Xtr_raw": Xtr_raw, "Xte_raw": Xte_raw,
        "Xtr": (Xtr_raw - mu) / sd, "Xte": (Xte_raw - mu) / sd,
        "ytr": data[tr, OUTCOME], "yte": data[te, OUTCOME],
        "mu": mu, "sd": sd,
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(X, y, eta, epochs, record=()):
    """Batch gradient descent from zero. Returns weights, bias and {epoch: training loss}."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses = {0: bce(y, sigmoid(X @ w + b))}
    for e in range(1, epochs + 1):
        err = sigmoid(X @ w + b) - y
        w = w - eta * (X.T @ err) / n
        b = b - eta * err.mean()
        if e in record:
            losses[e] = bce(y, sigmoid(X @ w + b))
    return w, b, losses


def confusion(y, pred):
    pred = np.asarray(pred).astype(bool)
    pos = np.asarray(y) == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    fn = int(np.sum(~pred & pos))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b else 0.0


def rates(tp, fp, tn, fn):
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "specificity": _div(tn, tn + fp),
        "fpr": _div(fp, fp + tn),
    }


def money_cost(tp, fp, tn, fn):
    """INR 50 per false alarm, INR 400 per missed churner."""
    return 50 * fp + 400 * fn


# ---------------------------------------------------------------- K-nearest neighbours
def knn_neighbor_labels(Xref, yref, Xq, kmax):
    """Labels of the kmax nearest reference rows for each query, nearest first.
    Ties in distance are broken by file order (stable sort)."""
    d = ((Xq[:, None, :] - Xref[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d, axis=1, kind="stable")[:, :kmax]
    return yref[order]


def knn_predict(labels, K):
    pos = labels[:, :K].sum(axis=1)
    return pos >= (K + 1) / 2, pos / K


KNN_KS = list(range(1, 22, 2))


def knn_cross_validate(Xtr, ytr, ks=KNN_KS, folds=5):
    fold = np.arange(len(ytr)) % folds
    scores = {K: [] for K in ks}
    for f in range(folds):
        ref, q = fold != f, fold == f
        labels = knn_neighbor_labels(Xtr[ref], ytr[ref], Xtr[q], max(ks))
        for K in ks:
            pred, _ = knn_predict(labels, K)
            scores[K].append(rates(*confusion(ytr[q], pred))["f1"])
    return [(K, float(np.mean(scores[K]))) for K in ks]


def best_k(cv):
    best = cv[0]
    for K, s in cv:
        if s > best[1]:
            best = (K, s)
    return best[0]
