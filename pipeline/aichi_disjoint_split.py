"""Shared helper: disjoint train/eval splits on the 3 Aichi maps.

Returns (train_scenes, eval_scenes). Guaranteed disjoint:
  - train uses seed=42 to sample n_train ego lanelets
  - eval uses seed=eval_seed sampled from REMAINING lanelets (after removing train ids)

aichi_hub_v2 has only 4 road lanelets, so we keep all 4 as eval there and use 0 train scenes from it (training uses ishio_dai + morikoro only).

BUG FIX (2026-05-17): an earlier version used Python's builtin hash(tag) for the per-map
salt, but hash(str) is randomised per-process (PYTHONHASHSEED), so the same
(train_seed, eval_seed) drew DIFFERENT lanelets across python invocations. We replace
it with zlib.crc32 which is deterministic across processes."""
import os, sys, random, zlib
sys.path.insert(0, '/opt/autoware/local/lib/python3.10/dist-packages')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner')
sys.path.insert(0, 'third_party/Diffusion-Planner/diffusion_planner_ros')
sys.path.insert(0, '.')

import numpy as np
import torch
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
from overnight_stage1_multiego import UNSEEN, build_inputs, SEED


def _stable_salt(tag: str) -> int:
    """Deterministic per-map salt; does not depend on PYTHONHASHSEED."""
    return zlib.crc32(tag.encode('utf-8')) % 10000


def disjoint_scene_split(n_train_per_map=30, n_eval_per_map=10,
                          train_seed=42, eval_seed=7):
    """Build disjoint train and eval scene lists across the 3 Aichi maps.

    Each scene is a dict with keys: map, cx, cy, heading, inputs.
    aichi_hub_v2 only has 4 lanelets -> all become eval scenes; no train from it.
    """
    train_scenes, eval_scenes = [], []
    for tag, osm in UNSEEN:
        try:
            llmap = convert_lanelet(osm)
        except Exception:
            continue
        road_lls = list(llmap.lanelets.values())
        N = len(road_lls)
        if N == 0: continue

        # Always reserve at least n_eval (or all available, whichever is smaller) for eval
        n_eval = min(n_eval_per_map, N)
        n_train = min(n_train_per_map, max(0, N - n_eval))

        # Use train_seed to pick the train set; eval_seed to pick from the remainder.
        # The per-map salt uses zlib.crc32 (deterministic across processes), NOT the
        # Python builtin hash() which is randomised by PYTHONHASHSEED.
        salt = _stable_salt(tag)
        rng_tr = random.Random(train_seed + salt)
        all_idx = list(range(N))
        rng_tr.shuffle(all_idx)
        train_idx = sorted(all_idx[:n_train])
        remaining = [i for i in all_idx if i not in set(train_idx)]
        rng_ev = random.Random(eval_seed + salt)
        rng_ev.shuffle(remaining)
        eval_idx = sorted(remaining[:n_eval])

        for split_name, idx_list, dest in [('train', train_idx, train_scenes),
                                            ('eval', eval_idx, eval_scenes)]:
            for i in idx_list:
                ego_ll = road_lls[i]
                cl = ego_ll.centerline
                if cl.shape[0] < 2: continue
                cx, cy = float(cl[0, 0]), float(cl[0, 1])
                heading = float(np.arctan2(cl[1, 1] - cl[0, 1], cl[1, 0] - cl[0, 0]))
                try:
                    inputs = build_inputs(llmap, (cx, cy), heading)
                    dest.append({'map': tag, 'cx': cx, 'cy': cy, 'heading': heading,
                                 'inputs': inputs, 'lanelet_idx': i, 'split': split_name})
                except Exception:
                    pass

    return train_scenes, eval_scenes


if __name__ == '__main__':
    tr, ev = disjoint_scene_split()
    print(f"train: {len(tr)}; eval: {len(ev)}")
    from collections import Counter
    print(f"train by map: {Counter(s['map'] for s in tr)}")
    print(f"eval  by map: {Counter(s['map'] for s in ev)}")
    # Verify disjoint
    tr_keys = {(s['map'], s['lanelet_idx']) for s in tr}
    ev_keys = {(s['map'], s['lanelet_idx']) for s in ev}
    print(f"disjoint: {len(tr_keys & ev_keys) == 0}")
