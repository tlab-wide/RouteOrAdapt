#!/usr/bin/env python3
"""
Stage 3: SCR — Scene-Conditioned Router.
Train a small MLP that maps scene features → expert ID on the 4 trained ODDs,
then evaluate it as a deployment-time router on the 3 unseen ODDs.

Scene features (cheap to extract):
  - n_lanes_in_view (0-140)
  - mean_lane_curvature
  - mean_lane_length_m
  - lateral_lane_spread (std of lane y midpoints)
  - longitudinal_lane_spread (std of lane x midpoints)
  - mean_neighbor_speed (we use 0 since dummy)

Training data: 6377 npz files; each labeled with its ODD expert ID.
"""
import os, sys, json, random, time
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ODD_LABELS = ['hongo', 'kashiwa', 'shalun', 'unagoya']  # 0..3
NUM_FEATS = 8


def features_from_npz(path):
    d = np.load(path)
    lanes = d['lanes']  # (140, 20, 33), x in feat 0, y in feat 1
    nb_pasts = d['neighbor_agents_past']  # (32, 31, 11), x/y in 0/1
    # Lane mask: ones if abs sum>0
    lane_mask = (np.abs(lanes[:, :, :2]).sum(axis=(1,2)) > 0.1).astype(float)
    n_lanes = float(lane_mask.sum())
    if n_lanes == 0:
        return np.zeros(NUM_FEATS, dtype=np.float32)

    # Lane mid points
    mids = lanes[lane_mask.astype(bool), :, :2].mean(axis=1)  # (n,2)
    lane_y_spread = float(mids[:, 1].std()) if mids.shape[0] > 1 else 0.0
    lane_x_spread = float(mids[:, 0].std()) if mids.shape[0] > 1 else 0.0
    lane_lengths = np.linalg.norm(lanes[lane_mask.astype(bool), -1, :2] - lanes[lane_mask.astype(bool), 0, :2], axis=1)
    mean_len = float(lane_lengths.mean())

    # Lane curvature: average heading change along each lane
    curvs = []
    for i in range(lanes.shape[0]):
        if not lane_mask[i]: continue
        pts = lanes[i, :, :2]
        if pts.shape[0] < 3: continue
        diffs = np.diff(pts, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        # absolute heading change
        dh = np.abs(np.diff(headings))
        curvs.append(float(np.mean(dh)))
    mean_curv = float(np.mean(curvs)) if curvs else 0.0

    # Neighbor speed
    nb_mask = (np.abs(nb_pasts[:, -1, :2]).sum(axis=1) > 0.1)
    nb_speed = 0.0
    if nb_mask.any() and nb_pasts.shape[1] >= 2:
        last2 = nb_pasts[nb_mask, -2:, :2]
        speeds = np.linalg.norm(last2[:, 1] - last2[:, 0], axis=1) / 0.1  # 10 Hz
        nb_speed = float(speeds.mean())

    # Number of route lanes
    route = d.get('route_lanes')
    n_route = 0.0
    if route is not None:
        n_route = float((np.abs(route[:, :, :2]).sum(axis=(1,2)) > 0.1).sum())

    return np.array([n_lanes, mean_curv, mean_len, lane_y_spread, lane_x_spread,
                     nb_speed, n_route, float(d.get('ego_current_state', np.zeros(10))[0])],
                    dtype=np.float32)


def get_odd(path):
    p = path.lower()
    for i, m in enumerate(ODD_LABELS):
        if m in p: return i
    return -1


class SCR(nn.Module):
    def __init__(self, n_in=NUM_FEATS, n_hid=64, n_out=len(ODD_LABELS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.GELU(),
            nn.Linear(n_hid, n_hid), nn.GELU(),
            nn.Linear(n_hid, n_out))
    def forward(self, x): return self.net(x)


def train_scr(X, y, epochs=50, lr=1e-3, bs=256, val_frac=0.1):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    # Split
    perm = torch.randperm(len(X))
    n_val = int(val_frac * len(X))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = X_t[tr_idx].to(DEVICE), y_t[tr_idx].to(DEVICE)
    Xval, yval = X_t[val_idx].to(DEVICE), y_t[val_idx].to(DEVICE)

    # Standardize
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-3)
    Xtr_n = (Xtr - mu) / sd
    Xval_n = (Xval - mu) / sd

    model = SCR().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_n))
        loss_sum = 0.0
        for i in range(0, len(Xtr_n), bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr_n[idx], ytr[idx]
            logits = model(xb)
            l = crit(logits, yb)
            opt.zero_grad(); l.backward(); opt.step()
            loss_sum += l.item() * len(idx)
        loss_sum /= len(Xtr_n)
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_n).argmax(-1)
            val_acc = float((val_pred == yval).float().mean())
        if val_acc > best_val_acc: best_val_acc = val_acc
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1:>3}: train_loss={loss_sum:.4f}  val_acc={val_acc:.3f}  best={best_val_acc:.3f}")
    return model, mu, sd, best_val_acc


def main():
    print("[stage3] loading npz path list...")
    paths = json.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    paths = [p for p in paths if get_odd(p) >= 0]  # filter to known 4 ODDs
    random.Random(42).shuffle(paths)
    paths = paths[:3000]  # 3000 random samples for tractable feature extraction
    print(f"  using {len(paths)} npz")

    print("[stage3] extracting features...")
    X = np.zeros((len(paths), NUM_FEATS), dtype=np.float32)
    y = np.zeros(len(paths), dtype=np.int64)
    t0 = time.time()
    fail = 0
    for i, p in enumerate(paths):
        try:
            X[i] = features_from_npz(p)
            y[i] = get_odd(p)
        except Exception as e:
            fail += 1
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s, {fail} failed)")
    print(f"  feature extraction done ({fail} failed)")

    # Class balance check
    from collections import Counter
    counts = Counter(y.tolist())
    print(f"  class counts: {dict(counts)}")

    print("[stage3] training SCR MLP...")
    model, mu, sd, val_acc = train_scr(X, y, epochs=80)
    print(f"  best val_acc = {val_acc:.3f}")

    # Save model
    out = {'description': 'Stage 3 SCR: scene-feature → expert classifier',
           'features': ['n_lanes','mean_curv','mean_len','lane_y_spread','lane_x_spread',
                        'nb_speed','n_route','ego_current_0'],
           'odd_labels': ODD_LABELS,
           'n_train_total': len(paths),
           'class_counts': {ODD_LABELS[k]: v for k, v in counts.items() if k >= 0},
           'val_accuracy': val_acc,
           'norm_mu': mu.cpu().numpy().tolist()[0],
           'norm_sd': sd.cpu().numpy().tolist()[0]}
    out_dir = 'checkpoints/overnight_20260510'
    json.dump(out, open(f'{out_dir}/stage3_scr.json', 'w'), indent=2)
    torch.save({'model': model.state_dict(), 'mu': mu.cpu(), 'sd': sd.cpu()},
               f'{out_dir}/stage3_scr.pth')
    print(f"  saved {out_dir}/stage3_scr.{{json,pth}}")


if __name__ == '__main__':
    main()
