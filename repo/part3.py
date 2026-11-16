"""Part 3: segmentation (k-means, hierarchical clustering) and PCA."""
import numpy as np

from common import OUTCOME, VALUE, load_data, write_csv, write_results

data = load_data()
X_raw = data[:, :5]
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
n = len(X)
R = []


def add(k, v):
    R.append((k, v))


# ---- 3.1 k-means from scratch with pinned starting rows
def kmeans(X, K):
    step = len(X) // K
    cent = X[[j * step for j in range(K)]].copy()
    labels, passes = None, 0
    while True:
        passes += 1
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)          # argmin returns the lowest index on ties
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(K):
            members = X[labels == j]
            if len(members):                 # an empty cluster keeps its centroid
                cent[j] = members.mean(axis=0)
    wcss = float(((X - cent[labels]) ** 2).sum())
    return labels, cent, wcss, passes


fits = {}
for K in range(2, 7):
    fits[K] = kmeans(X, K)
    add(f"wcss_k{K}", fits[K][2])

# ---- 3.2 elbow rule
W = {K: fits[K][2] for K in fits}
elbow = None
for K in (3, 4, 5):
    if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
        elbow = K
        break
add("elbow_k", elbow if elbow is not None else 0)

# ---- 3.3 segments for K = 3, ordered by mean discount share (X3)
labels3 = fits[3][0]
order = sorted(range(3), key=lambda j: X_raw[labels3 == j, 2].mean())
seg_of = np.empty(n, dtype=int)
rows = []
for name, j in zip("ABC", order):
    m = labels3 == j
    seg_of[m] = "ABC".index(name)
    rows.append([name, int(m.sum())] + [X_raw[m, c].mean() for c in range(5)]
                + [data[m, OUTCOME].mean(), data[m, VALUE].mean()])
    add(f"size_segment_{name}", int(m.sum()))
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
          rows)

# ---- 3.4 hierarchical clustering on the supplied table
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def link(a, b, rule):
    d = [D[i, j] for i in a for j in b]
    return min(d) if rule == "single" else max(d) if rule == "complete" else sum(d) / len(d)


def label(c):
    return "+".join(names[i] for i in sorted(c))


def agglomerate(rule):
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                h = link(clusters[a], clusters[b], rule)
                if best is None or h < best[0]:
                    best = (h, a, b)
        h, a, b = best
        ca, cb = clusters[a], clusters[b]
        first, second = (ca, cb) if min(ca) < min(cb) else (cb, ca)
        merges.append((label(first), label(second), h))
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [sorted(ca + cb)]
    return merges


hier_rows = []
for rule in ("single", "complete", "average"):
    merges = agglomerate(rule)
    for i, (a, b, h) in enumerate(merges, 1):
        hier_rows.append([rule, i, a, b, h])
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = int(np.argmax(gaps))                 # gap g lies between merge g+1 and merge g+2
    add(f"{rule}_clusters_at_cut", 6 - (g + 1))
write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)

# ---- 3.5 PCA in seven steps
mean_vec = X.mean(axis=0)                    # step 1 (all ~0 after standardising)
Xc = X - mean_vec                            # step 2
S = (Xc.T @ Xc) / (n - 1)                    # step 3
vals, vecs = np.linalg.eigh(S)               # step 4
idx = np.argsort(vals)[::-1]                 # step 5
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):                           # sign convention
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
Wk = vecs[:, :2]                             # step 6
Z = Xc @ Wk                                  # step 7
total = vals.sum()
add("eigenvalue_sum", total)
for j in range(5):
    add(f"eigenvalue_{j + 1}", vals[j])
ratio = vals / total
for j in range(5):
    add(f"explained_ratio_{j + 1}", ratio[j])
cum = np.cumsum(ratio)
for j in range(5):
    add(f"cumulative_ratio_{j + 1}", cum[j])
add("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for c in (1, 2):
    for j in range(5):
        add(f"pc{c}_loading_X{j + 1}", vecs[j, c - 1])
for r in range(3):
    add(f"row{r + 1}_pc1", Z[r, 0])
    add(f"row{r + 1}_pc2", Z[r, 1])
for s, name in enumerate("ABC"):
    add(f"mean_pc1_segment_{name}", Z[seg_of == s, 0].mean())

write_results("part3_results.txt", R)
