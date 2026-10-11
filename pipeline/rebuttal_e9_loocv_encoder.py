#!/usr/bin/env python3
"""E9 (rebuttal): LOOCV oracle-expert prediction from the 256-d planner-encoder
embedding (the representation the KSG/MI analysis used), completing E3 which
used the 8-d hand-crafted features. Answers 4EkA W2 / Nkac Q6 fully:
neither hand-crafted nor learned scene representations beat the majority baseline."""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')
sys.path.insert(0, '.')

import numpy as np
from diffusion_planner.utils.config import Config
from overnight_stage1_multiego import ARGS
from encoder_feature_mi_conformal import load_base_model, gather_aichi_scenes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
cfg = Config(ARGS)
model = load_base_model(cfg)
rows = gather_aichi_scenes(model, cfg)
print(f'{len(rows)} scenes with embeddings')

olab = {(r['map'], r['ego_idx']): r['oracle']
        for r in json.load(open('outputs/oracle_pool_ood_backup.json'))}
X = np.array([r['enc'] for r in rows])
y = np.array([olab[(r['map'], r['ego_idx'])] for r in rows])
n = len(y)
counts = {c: int((y == c).sum()) for c in sorted(set(y))}
majority = max(counts.values()) / n
print(f'n={n} d={X.shape[1]} counts={counts} majority={majority:.3f}')

def loocv_acc(mk, Xm, ym):
    correct = 0
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        clf = mk(); clf.fit(Xm[m], ym[m])
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
    null = np.array([loocv_acc(mk, X, rng.permutation(y)) for _ in range(200)])
    p = float((null >= acc).mean())
    res[name] = dict(loocv_acc=acc, perm_p=p, null_mean=float(null.mean()))
    print(f'{name:12s} LOOCV acc={acc:.3f}  perm-p={p:.3f}  null_mean={null.mean():.3f}')

json.dump(dict(description=__doc__, n=n, d=int(X.shape[1]),
               class_counts=counts, majority_baseline=majority, results=res),
          open('outputs/rebuttal/e9_loocv_encoder.json', 'w'), indent=1)
print('saved e9_loocv_encoder.json')
