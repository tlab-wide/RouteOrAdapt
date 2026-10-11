"""§D + §E: MI/Fano + conformal calibration with planner-encoder features.

Replaces the 8-d hand-crafted feature representation used by the routing
baselines, the MI analysis (§5.routing_explanation), and the split-conformal
predictor (§5.conformal) with a 256-d mean-pooled encoder embedding from the
Diffusion-Planner encoder. Tests whether:
  (D) the Fano-bound conclusion ("no scene features carry information about
      the per-scene oracle expert on OOD") survives a richer representation;
  (E) the split-conformal exchangeability rejection (p=0.0035) persists.

Output: outputs/encoder_feature_mi_conformal.json
"""
import os, sys, json, time, random
os.environ.setdefault('LD_LIBRARY_PATH', '/opt/autoware/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, 'third_party/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '.')

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet

from utils import load_npz_data
from overnight_stage1_multiego import (build_inputs, pick_n_egos, UNSEEN, N_EGO, SEED, ARGS, BASE_PTH)
from overnight_stage11_conformal import features_from_inputs

DEVICE = torch.device('cuda')
OUT = 'outputs/encoder_feature_mi_conformal.json'
EXPERTS = ['hongo', 'kashiwa', 'shalun', 'unagoya']
S1 = 'checkpoints/overnight_20260510/stage1_multiego.json'


def load_base_model(cfg):
    model = Diffusion_Planner(cfg).to(DEVICE).eval()
    ckpt = torch.load(BASE_PTH, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('ema_state_dict') or ckpt['model']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def encoder_embedding(model, cfg, inputs):
    """Run the encoder, return mean-pooled (1, hidden_dim) embedding."""
    inputs_norm = cfg.observation_normalizer({
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()})
    enc_out = model.encoder(inputs_norm)
    if isinstance(enc_out, dict):
        feat = enc_out.get('encoding') or enc_out.get('memory') or list(enc_out.values())[0]
    else:
        feat = enc_out
    if isinstance(feat, tuple):
        feat = feat[0]
    pooled = feat.mean(dim=1).squeeze(0).cpu().numpy()  # (hidden_dim,)
    return pooled


def gather_aichi_scenes(model, cfg):
    """For each Aichi map, generate scenes and extract encoder embeddings."""
    rows = []
    for tag, osm in UNSEEN:
        try:
            llmap = convert_lanelet(osm)
        except Exception as e:
            print(f"  skip {tag}: {e}"); continue
        egos = pick_n_egos(llmap, N_EGO, seed=SEED)
        for ego_idx, ego_ll in enumerate(egos):
            cl = ego_ll.centerline
            if cl.shape[0] < 2: continue
            cx, cy = float(cl[0, 0]), float(cl[0, 1])
            heading = float(np.arctan2(cl[1, 1]-cl[0, 1], cl[1, 0]-cl[0, 0]))
            try:
                inputs = build_inputs(llmap, (cx, cy), heading)
                enc_feat = encoder_embedding(model, cfg, inputs)
                hand_feat = features_from_inputs(inputs)
                rows.append(dict(map=tag, ego_idx=ego_idx,
                                 enc=enc_feat.tolist(), hand=hand_feat.tolist()))
            except Exception as e:
                print(f"  skip {tag} ego{ego_idx}: {e}")
    return rows


def gather_prod_scenes(model, cfg, npz_paths, n_per_odd=12):
    """For each prod ODD, sample N npz files and extract encoder embeddings.
    The 'oracle expert' for prod scenes is the matched ODD."""
    rows = []
    PROD_ODDS = EXPERTS
    per_odd = {m: [p for p in npz_paths if m in p.lower()] for m in PROD_ODDS}
    rng = random.Random(2026)
    for odd_idx, odd in enumerate(PROD_ODDS):
        pool = per_odd[odd]; rng.shuffle(pool)
        for p in pool[:n_per_odd]:
            try:
                data = load_npz_data(p, DEVICE)
                enc_feat = encoder_embedding(model, cfg, data)
                hand_feat = features_from_inputs(data)
                rows.append(dict(odd=odd, oracle_idx=odd_idx,
                                 enc=enc_feat.tolist(), hand=hand_feat.tolist()))
            except Exception as e:
                continue
    return rows


def attach_per_expert_lats(aichi_rows):
    """Stage1 multiego has per-expert max_lat for each Aichi (map, ego_idx)."""
    s1 = json.load(open(S1))
    lats = {}
    for r in s1['rows']:
        if not r.get('finite'): continue
        if r['expert'] not in EXPERTS: continue
        lats.setdefault((r['map'], r['ego_idx']), {})[r['expert']] = float(r['max_lat'])
    for r in aichi_rows:
        key = (r['map'], r['ego_idx'])
        if key in lats and all(e in lats[key] for e in EXPERTS):
            r['lats'] = [lats[key][e] for e in EXPERTS]
            r['oracle_idx'] = int(np.argmin(r['lats']))
        else:
            r['lats'] = None
    return [r for r in aichi_rows if r['lats'] is not None]


def main():
    import json as _j
    cfg = Config(ARGS, guidance_fn=None)
    print(f"[§D+E] Loading base model (no LoRA)...")
    model = load_base_model(cfg)

    print(f"[§D+E] Gathering Aichi encoder embeddings...")
    aichi_rows = gather_aichi_scenes(model, cfg)
    aichi_rows = attach_per_expert_lats(aichi_rows)
    print(f"  Aichi rows with lats: {len(aichi_rows)}")

    # prod-ODD scenes: load path_list_train_v3, take 12 per ODD
    print(f"[§D+E] Gathering prod-ODD encoder embeddings...")
    prod_paths = _j.load(open('checkpoints/npz_multimap/path_list_train_v3.json'))
    prod_rows = gather_prod_scenes(model, cfg, prod_paths, n_per_odd=12)
    print(f"  prod rows: {len(prod_rows)}")

    # ── §D: MI/Fano on (a) hand-crafted 8-d features, (b) encoder 256-d ──
    def run_mi(X, y, label):
        """Returns KSG-MI estimate + permutation null."""
        rng = np.random.default_rng(2026)
        mi_obs = mutual_info_classif(X, y, n_neighbors=3, random_state=2026).sum()
        nulls = []
        for _ in range(500):
            yp = rng.permutation(y)
            nulls.append(mutual_info_classif(X, yp, n_neighbors=3, random_state=2026).sum())
        nulls = np.array(nulls)
        p_one_sided = float(np.mean(nulls >= mi_obs))
        return dict(label=label, mi_observed=float(mi_obs),
                    null_mean=float(nulls.mean()), null_p95=float(np.percentile(nulls, 95)),
                    p_value=p_one_sided, n=len(y), d=X.shape[1])

    X_aichi_hand = np.array([r['hand'] for r in aichi_rows])
    X_aichi_enc  = np.array([r['enc']  for r in aichi_rows])
    y_aichi      = np.array([r['oracle_idx'] for r in aichi_rows])
    print(f"  Aichi shapes: hand={X_aichi_hand.shape} enc={X_aichi_enc.shape} y={y_aichi.shape}")

    mi_hand_aichi = run_mi(X_aichi_hand, y_aichi, 'aichi-hand-8d')
    mi_enc_aichi  = run_mi(X_aichi_enc, y_aichi, 'aichi-encoder-256d')

    # On prod we can also evaluate MI: the oracle is the matched ODD (a priori bijection
    # with the ODD label, so MI is theoretically high — useful as a sanity check)
    X_prod_hand = np.array([r['hand'] for r in prod_rows])
    X_prod_enc  = np.array([r['enc']  for r in prod_rows])
    y_prod      = np.array([r['oracle_idx'] for r in prod_rows])
    mi_hand_prod = run_mi(X_prod_hand, y_prod, 'prod-hand-8d')
    mi_enc_prod  = run_mi(X_prod_enc, y_prod, 'prod-encoder-256d')

    # ── §E: conformal predictor on encoder features ──
    # Replicate the prod-vs-aichi exchangeability test using encoder features.
    # Use a Ridge regressor from the encoder features to a target proxy: we use
    # the per-scene oracle lateral offset as the regression target for Aichi,
    # and a constant 4.0 m for prod (in-distribution well-trained baseline) as
    # the regression target for prod. The conformal residual is |y - ŷ|; the
    # permutation test compares prod-cal residuals vs aichi residuals.
    def conformal_exchangeability_test(X_cal_prod, y_cal_prod, X_aichi, y_aichi, n_perm=10000):
        Xtr = X_cal_prod; ytr = y_cal_prod
        # 60-40 split prod
        n = len(Xtr); idx = np.arange(n); np.random.default_rng(42).shuffle(idx)
        ntr = int(0.6 * n)
        Xfit, yfit = Xtr[idx[:ntr]], ytr[idx[:ntr]]
        Xcal, ycal = Xtr[idx[ntr:]], ytr[idx[ntr:]]
        reg = Ridge(alpha=1.0).fit(Xfit, yfit)
        res_cal = np.abs(ycal - reg.predict(Xcal))
        res_aichi = np.abs(y_aichi - reg.predict(X_aichi))
        # Two-sided permutation test of difference of means
        obs_diff = res_aichi.mean() - res_cal.mean()
        combined = np.concatenate([res_cal, res_aichi])
        n_cal = len(res_cal); n_aichi = len(res_aichi)
        rng = np.random.default_rng(2026)
        nulls = []
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            d = perm[n_cal:].mean() - perm[:n_cal].mean()
            nulls.append(d)
        nulls = np.array(nulls)
        p_two = float(np.mean(np.abs(nulls) >= abs(obs_diff)))
        return dict(res_cal_mean=float(res_cal.mean()), res_aichi_mean=float(res_aichi.mean()),
                    obs_diff=float(obs_diff), p_value=p_two,
                    n_cal=n_cal, n_aichi=n_aichi)

    # For conformal we use the per-scene oracle lat as the regression target
    # (this approximates the realised lateral offset of the matched expert).
    y_aichi_target = np.array([min(r['lats']) for r in aichi_rows])
    y_prod_target  = np.full(len(prod_rows), 4.0)  # in-distribution well-trained baseline
    conf_hand = conformal_exchangeability_test(X_prod_hand, y_prod_target, X_aichi_hand, y_aichi_target)
    conf_enc  = conformal_exchangeability_test(X_prod_enc, y_prod_target, X_aichi_enc, y_aichi_target)

    # ── §F: per-feature MI ablation on hand-crafted features ──
    f_mi = []
    for j in range(X_aichi_hand.shape[1]):
        Xj = X_aichi_hand[:, j:j+1]
        r = run_mi(Xj, y_aichi, f'aichi-hand-d{j}')
        f_mi.append(r)

    out = dict(
        description='§D+E+F: encoder-feature MI/conformal vs hand-crafted',
        n_aichi=len(aichi_rows), n_prod=len(prod_rows),
        hidden_dim=cfg.hidden_dim if hasattr(cfg, 'hidden_dim') else 256,
        D_mi=[mi_hand_aichi, mi_enc_aichi, mi_hand_prod, mi_enc_prod],
        E_conformal=dict(hand=conf_hand, encoder=conf_enc),
        F_per_feature_mi=f_mi,
    )
    json.dump(out, open(OUT, 'w'), indent=2, default=float)
    print(f"\nSaved {OUT}")
    print("\n=== §D: MI on Aichi oracle expert ===")
    print(f"  hand 8-d:    MI={mi_hand_aichi['mi_observed']:.3f}  p={mi_hand_aichi['p_value']:.3f}")
    print(f"  encoder 256: MI={mi_enc_aichi['mi_observed']:.3f}   p={mi_enc_aichi['p_value']:.3f}")
    print("\n=== §D: MI on prod oracle expert (sanity, should be HIGH) ===")
    print(f"  hand 8-d:    MI={mi_hand_prod['mi_observed']:.3f}  p={mi_hand_prod['p_value']:.3f}")
    print(f"  encoder 256: MI={mi_enc_prod['mi_observed']:.3f}   p={mi_enc_prod['p_value']:.3f}")
    print("\n=== §E: conformal residual exchangeability ===")
    print(f"  hand 8-d:    res_aichi={conf_hand['res_aichi_mean']:.2f} vs res_cal={conf_hand['res_cal_mean']:.2f}  p={conf_hand['p_value']:.4f}")
    print(f"  encoder 256: res_aichi={conf_enc['res_aichi_mean']:.2f} vs res_cal={conf_enc['res_cal_mean']:.2f}  p={conf_enc['p_value']:.4f}")
    print("\n=== §F: per-feature MI (hand-crafted 8 features) ===")
    f_names = ['n_lanes', 'lane_y_spread', 'lane_x_spread', 'mean_len',
                'mean_curv', 'n_neighbors', 'mean_nb_speed', 'route_lane_count']
    for j, r in enumerate(f_mi):
        nm = f_names[j] if j < len(f_names) else f'd{j}'
        print(f"  [{j}] {nm:<22s}  MI={r['mi_observed']:.4f}  p={r['p_value']:.3f}")


if __name__ == '__main__':
    main()
