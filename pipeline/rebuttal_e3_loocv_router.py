#!/usr/bin/env python3
"""E3 (rebuttal): replace the flawed Fano/summed-MI argument with a direct,
assumption-free test — leave-one-out cross-validated classifiers predicting the
per-scene oracle expert (4-way) from scene features, on the 24 Pool-A scenes.

If LOOCV accuracy is at/below the majority-class baseline, scene features carry
no usable routing signal — the clean replacement for the KSG/Fano analysis.
Answers 4EkA W2 and Nkac Q6.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)

feat_rows = json.load(open('checkpoints/overnight_20260513/stage11_conformal_v3.json'))['aichi_scenes']
oracle_rows = json.load(open('outputs/oracle_pool_ood_backup.json'))
olab = {(r['map'], r['ego_idx']): r['oracle'] for r in oracle_rows}

X, y, keys = [], [], []
for r in feat_rows:
    k = (r['map'], r['ego_idx'])
    X.append(r['features']); y.append(olab[k]); keys.append(k)
X = np.array(X); classes = sorted(set(y)); y = np.array(y)
n = len(y)
assert n == 24 and X.shape[1] == 8

counts = {c: int((y == c).sum()) for c in classes}
majority = max(counts.values()) / n

def loocv_acc(mk, Xm, ym):
    correct = 0
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        clf = mk()
        clf.fit(Xm[m], ym[m])
        correct += int(clf.predict(Xm[i:i+1])[0] == ym[i])
    return correct / n

models = {
    'logistic_L2': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
    '1NN': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
    '3NN': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    'rbf_SVM': lambda: make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0)),
}

res = {}
for name, mk in models.items():
    acc = loocv_acc(mk, X, y)
    # permutation test: LOOCV accuracy under label permutation (500 perms)
    null = []
    for _ in range(500):
        yp = rng.permutation(y)
        null.append(loocv_acc(mk, X, yp))
    null = np.array(null)
    p = float((null >= acc).mean())
    res[name] = dict(loocv_acc=acc, n_correct=int(round(acc * n)),
                     perm_p=p, null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)))
    print(f"{name:12s} LOOCV acc = {acc:.3f} ({int(round(acc*n))}/24)  perm-p = {p:.3f}  "
          f"null mean {null.mean():.3f}, null p95 {np.percentile(null,95):.3f}")

out = dict(description=__doc__, n=n, d=X.shape[1],
           class_counts=counts, chance_uniform=0.25, majority_baseline=majority,
           results=res)
print(f"\nmajority-class baseline = {majority:.3f}  (counts: {counts})")
with open('outputs/rebuttal/e3_loocv_router.json', 'w') as f:
    json.dump(out, f, indent=1)
