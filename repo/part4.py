"""Part 4: split purity, model combination, AdaBoost by hand, OOB share, network sizing."""
import math

import numpy as np

from common import (KNN_KS, best_k, confusion, knn_cross_validate, knn_neighbor_labels, knn_predict,
                    load_data, money_cost, prepare, rates, sigmoid, train_logistic, write_csv,
                    write_results)

data = load_data()
S = prepare(data)
Xtr_raw, Xte_raw = S["Xtr_raw"], S["Xte_raw"]
Xtr, Xte, ytr, yte = S["Xtr"], S["Xte"], S["ytr"], S["yte"]
R = []


def add(k, v):
    R.append((k, v))


# ---- 4.1 which split is purest?
def entropy(pos, n):
    if n == 0:
        return 0.0
    out = 0.0
    for p in (pos / n, 1 - pos / n):
        if p > 0:
            out -= p * math.log2(p)
    return out


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p ** 2 - (1 - p) ** 2


N, P = len(ytr), int(ytr.sum())
root_e, root_g = entropy(P, N), gini(P, N)
add("root_entropy", root_e)
add("root_gini", root_g)

splits = {
    "S1": lambda X: X[:, 5] == 1,        # cod_flag = 1
    "S2": lambda X: X[:, 6] == 1,        # plus_member_flag = 1
    "S3": lambda X: X[:, 2] > 0.5,       # discount_order_share > 0.5
    "S4": lambda X: X[:, 0] <= 24,       # tenure <= 24 months
}
rows, ig, gg = [], {}, {}
for name, f in splits.items():
    yes = f(Xtr_raw)
    n_yes, p_yes = int(yes.sum()), int(ytr[yes].sum())
    n_no, p_no = N - n_yes, P - p_yes
    ig[name] = root_e - (n_yes / N * entropy(p_yes, n_yes) + n_no / N * entropy(p_no, n_no))
    gg[name] = root_g - (n_yes / N * gini(p_yes, n_yes) + n_no / N * gini(p_no, n_no))
    rows.append([name, n_yes, p_yes, n_no, p_no, ig[name], gg[name]])
write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows)
best_ig = max(ig, key=lambda k: ig[k])
best_gg = max(gg, key=lambda k: gg[k])
add("best_split_by_info_gain", best_ig)
add("best_split_by_gini_gain", best_gg)

# ---- 4.2 stump, KNN, logistic and two ways to combine them
yes_tr = splits[best_ig](Xtr_raw)
rate_yes, rate_no = float(ytr[yes_tr].mean()), float(ytr[~yes_tr].mean())
add("stump_rate_yes", rate_yes)
add("stump_rate_no", rate_no)

w, b, _ = train_logistic(Xtr, ytr, 0.1, 1000)
p_log = sigmoid(Xte @ w + b)
K = best_k(knn_cross_validate(Xtr, ytr))
labels = knn_neighbor_labels(Xtr, ytr, Xte, max(KNN_KS))
pred_knn, p_knn = knn_predict(labels, K)
p_stump = np.where(splits[best_ig](Xte_raw), rate_yes, rate_no)

pred = {
    "logistic": p_log >= 0.3,
    "knn": pred_knn,
    "stump": p_stump >= 0.3,
}
pred["hard_vote"] = (pred["logistic"].astype(int) + pred["knn"].astype(int) + pred["stump"].astype(int)) >= 2
pred["soft_vote"] = (p_log + p_knn + p_stump) / 3 >= 0.3
vote_rows = []
for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    tp, fp, tn, fn = confusion(yte, pred[name])
    r = rates(tp, fp, tn, fn)
    vote_rows.append([name, tp, fp, tn, fn, r["precision"], r["recall"], r["f1"]])
    # extra (not required): the Task 2.8 money rule applied to every model, used in the memo
    add(f"cost_test_{name}", money_cost(tp, fp, tn, fn))
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vote_rows)

# ---- 4.3 two rounds of AdaBoost by hand
y8 = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h = [np.array([1, 1, -1, -1, -1, -1, -1, -1]), np.array([-1, 1, 1, -1, 1, -1, 1, -1])]
w_t = np.full(8, 1 / 8)
alphas, ada_rows = [], []
for t in range(2):
    eps = float(w_t[h[t] != y8].sum())
    alpha = 0.5 * math.log((1 - eps) / eps)
    raw = w_t * np.exp(-alpha * y8 * h[t])
    new = raw / raw.sum()
    for i in range(8):
        ada_rows.append([t + 1, i + 1, w_t[i], raw[i], new[i]])
    add(f"round{t + 1}_error", eps)
    add(f"round{t + 1}_alpha", alpha)
    alphas.append(alpha)
    if t == 0:
        wrong_weight_after_round1 = float(new[h[0] != y8].sum())
    w_t = new
write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
score = alphas[0] * h[0] + alphas[1] * h[1]
add("ensemble_accuracy", float(np.mean(np.sign(score) == y8)))
add("round1_wrong_weight_after_normalisation", wrong_weight_after_round1)   # extra, quoted in README

# ---- 4.4 out-of-bag share
add("oob_fraction", (1 - 1 / 560) ** 560)

# ---- 4.5 sizing the networks (nothing is trained)
def conv_out(size, k, pad, stride=1):
    return (size - k + 2 * pad) // stride + 1


mlp = {"mlp_hidden1": 784 * 16 + 16, "mlp_hidden2": 16 * 16 + 16, "mlp_output": 16 * 10 + 10}
s1 = conv_out(28, 5, 2)          # conv1 -> 28
p1 = conv_out(s1, 2, 0, 2)       # pool  -> 14
s2 = conv_out(p1, 5, 0)          # conv2 -> 10
p2 = conv_out(s2, 2, 0, 2)       # pool  -> 5
flat = p2 * p2 * 16
cnn = {"cnn_conv1": 5 * 5 * 1 * 6 + 6, "cnn_conv2": 5 * 5 * 6 * 16 + 16,
       "cnn_dense1": flat * 32 + 32, "cnn_dense2": 32 * 10 + 10}
write_csv("part4_architecture.csv", ["layer", "parameters"], list({**mlp, **cnn}.items()))
add("mlp_total", sum(mlp.values()))
add("cnn_total", sum(cnn.values()))
add("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
add("conv_layer_params", 3 * 3 * 3 * 64 + 64)
add("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)
add("cnn_shapes_check", f"conv1 {s1}x{s1}x6, pool1 {p1}x{p1}x6, conv2 {s2}x{s2}x16, pool2 {p2}x{p2}x16, flatten {flat}")

write_results("part4_results.txt", R)
