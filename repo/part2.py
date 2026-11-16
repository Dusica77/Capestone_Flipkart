"""Part 2: classification (logistic regression + KNN) and evaluation."""
import numpy as np

from common import (KNN_KS, best_k, bce, confusion, knn_cross_validate, knn_neighbor_labels,
                    knn_predict, load_data, money_cost, prepare, rates, sigmoid, train_logistic,
                    write_csv, write_results)

data = load_data()
S = prepare(data)
Xtr, Xte, ytr, yte = S["Xtr"], S["Xte"], S["ytr"], S["yte"]
R = []


def add(k, v):
    R.append((k, v))


# ---- 2.1 split
add("train_rows", len(ytr))
add("train_positives", int(ytr.sum()))
add("test_rows", len(yte))
add("test_positives", int(yte.sum()))

# ---- 2.2 logistic regression from scratch
w, b, losses = train_logistic(Xtr, ytr, 0.1, 1000, record=(1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    add(f"loss_epoch_{e}", losses[e])
for j in range(7):
    add(f"weight_X{j + 1}", w[j])
add("bias", b)
p_tr = sigmoid(Xtr @ w + b)
p_te = sigmoid(Xte @ w + b)
add("test_bce", bce(yte, p_te))

# ---- 2.3 learning-rate effect
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    add(f"loss_epoch_200_eta_{eta}", l[200])

# ---- 2.4 odds ratios
for j in range(7):
    add(f"odds_ratio_X{j + 1}", np.exp(w[j]))

# ---- 2.5 two thresholds on the test rows
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, p_te >= tau)
    r = rates(tp, fp, tn, fn)
    for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
        add(f"tau_{tau}_{k}", v)
    for k in ("accuracy", "precision", "recall", "f1", "specificity"):
        add(f"tau_{tau}_{k}", r[k])
add("blind_accuracy", float(np.mean(yte == 0)))

# ---- 2.6 threshold sweep
rows = []
for i in range(1, 10):
    tau = i / 10
    tp, fp, tn, fn = confusion(yte, p_te >= tau)
    r = rates(tp, fp, tn, fn)
    rows.append([tau, tp, fp, tn, fn, r["recall"], r["fpr"]])
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# ---- 2.7 KNN with 5-fold CV
cv = knn_cross_validate(Xtr, ytr)
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[K, s] for K, s in cv])
K = best_k(cv)
add("knn_best_k", K)
labels = knn_neighbor_labels(Xtr, ytr, Xte, max(KNN_KS))
pred, _ = knn_predict(labels, K)
tp, fp, tn, fn = confusion(yte, pred)
r = rates(tp, fp, tn, fn)
for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
    add(f"knn_{k}", v)
for k in ("accuracy", "precision", "recall", "f1"):
    add(f"knn_{k}", r[k])

# ---- 2.8 money-based threshold, chosen on TRAINING rows
best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20
    c = money_cost(*confusion(ytr, p_tr >= tau))
    if best_cost is None or c < best_cost:
        best_tau, best_cost = tau, c
add("cost_tau", best_tau)
add("cost_train_at_tau", best_cost)
add("cost_test_at_tau", money_cost(*confusion(yte, p_te >= best_tau)))
add("cost_test_at_0.5", money_cost(*confusion(yte, p_te >= 0.5)))
add("cost_test_flag_nobody", money_cost(*confusion(yte, np.zeros(len(yte)))))

write_results("part2_results.txt", R)
