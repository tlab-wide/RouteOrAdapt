#!/usr/bin/env python3
"""E18: expand the routing-study pool from 24 to ~104 scenes (all 4 aichi_hub
lanelets + 50 ishio_dai + 50 morikoro, pick_n_egos seed=42, same protocol as
stage1). For every scene, roll out the four ODD experts + shared and compute:
  - static rule (unagoya) fleet mean
  - per-scene oracle fleet mean  -> oracle headroom at 4x the pool size
  - 8-d scene features + oracle labels -> LOOCV classifier with real power
Directly answers Nkac W1 (only 24 scenes) for the load-bearing claim."""
import os, sys, json
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from collections import Counter
from scipy import stats
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from train_reward_backprop import differentiable_dpm_solver_sample
from train_molora import set_active_expert, ODD_MAP_TO_IDX
from overnight_stage1_multiego import build_inputs, pick_n_egos, metrics_for_traj, ARGS, UNSEEN
from overnight_stage11_conformal import features_from_inputs
from rebuttal_e12_matched_baselines import build_model

EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
N_PER_MAP = {'aichi_hub_v2': 4, 'ishio_dai': 50, 'morikoro': 50}
DEVICE = torch.device('cuda')
OUT = 'outputs/rebuttal/e18_expanded_pool.json'

cfg = Config(ARGS); ns = NoiseScheduleVP()
model = build_model(cfg)
model.eval()

rows = []
for tag, osm in UNSEEN:
    llmap = convert_lanelet(osm)
    egos = pick_n_egos(llmap, N_PER_MAP[tag], seed=42)
    print(f'{tag}: {len(egos)} egos', flush=True)
    for ego_idx, ego_ll in enumerate(egos):
        cl = ego_ll.centerline
        if cl.shape[0] < 2:
            continue
        cx, cy = float(cl[0, 0]), float(cl[0, 1])
        heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
        try:
            inputs = build_inputs(llmap, (cx, cy), heading)
            feats = [float(x) for x in features_from_inputs(inputs)]
            inputs_norm = cfg.observation_normalizer(
                {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
        except Exception as e:
            continue
        row = dict(map=tag, ego_idx=ego_idx, lanelet_id=int(ego_ll.id), features=feats)
        ok = True
        for e in EXPERTS + ['shared']:
            idx = ODD_MAP_TO_IDX[e] if e in ODD_MAP_TO_IDX else None
            set_active_expert(model, idx)
            try:
                with torch.no_grad():
                    tj = differentiable_dpm_solver_sample(model, cfg, inputs_norm, ns, device=DEVICE)
                m = metrics_for_traj(tj[:, :2].cpu().numpy())
                if not m.get('finite'):
                    ok = False; break
                row[e] = m['max_lat']
            except Exception:
                ok = False; break
        if ok:
            rows.append(row)
        if len(rows) % 20 == 0 and ok:
            print(f'  {len(rows)} scenes done', flush=True)

n = len(rows)
static = np.array([r['unagoya'] for r in rows])
oracle = np.array([min(r[e] for e in EXPERTS) for r in rows])
d = static - oracle
w = stats.wilcoxon(d[d > 1e-9])
boot = [np.mean(rng_d) for rng_d in (np.random.default_rng(0).choice(d, size=(2000, n), replace=True))]
lo, hi = np.percentile(boot, [2.5, 97.5])
olabels = [min(EXPERTS, key=lambda e: r[e]) for r in rows]
cnt = Counter(olabels)
maj = max(cnt.values()) / n

print(f'\n=== EXPANDED POOL n={n} ===')
print(f'static (unagoya) fleet mean: {static.mean():.2f}')
print(f'oracle fleet mean:           {oracle.mean():.2f}')
print(f'ORACLE HEADROOM:             {d.mean():.2f} m  (95% CI [{lo:.2f}, {hi:.2f}])  Wilcoxon p={w.pvalue:.2e}')
print(f'oracle label counts: {dict(cnt)}  majority baseline={maj:.3f}')
per_exp = {e: float(np.mean([r[e] for r in rows])) for e in EXPERTS + ['shared']}
print('per-expert fleet means:', {k: round(v, 2) for k, v in per_exp.items()})

# LOOCV at real power
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X = np.array([r['features'] for r in rows]); y = np.array(olabels)
rng = np.random.default_rng(0)
loocv_res = {}
for name, mk in [('logistic', lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
                 ('3NN', lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(3))),
                 ('rbf_SVM', lambda: make_pipeline(StandardScaler(), SVC()))]:
    correct = 0
    for i in range(n):
        m_ = np.ones(n, bool); m_[i] = False
        c = mk(); c.fit(X[m_], y[m_]); correct += int(c.predict(X[i:i+1])[0] == y[i])
    acc = correct / n
    null = []
    for _ in range(200):
        yp = rng.permutation(y); corr = 0
        for i in range(n):
            m_ = np.ones(n, bool); m_[i] = False
            c = mk(); c.fit(X[m_], yp[m_]); corr += int(c.predict(X[i:i+1])[0] == yp[i])
        null.append(corr / n)
    p = float((np.array(null) >= acc).mean())
    loocv_res[name] = dict(acc=acc, perm_p=p)
    print(f'LOOCV {name}: acc={acc:.3f} (majority {maj:.3f})  perm-p={p:.3f}', flush=True)

json.dump(dict(description=__doc__, n=n, rows=rows,
               static_mean=float(static.mean()), oracle_mean=float(oracle.mean()),
               headroom=float(d.mean()), headroom_ci=[float(lo), float(hi)],
               wilcoxon_p=float(w.pvalue), oracle_counts=dict(cnt),
               majority=maj, per_expert=per_exp, loocv=loocv_res),
          open(OUT, 'w'), indent=1)
print('saved', OUT)
