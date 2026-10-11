"""Rebuild the checkpoint files the evaluation scripts expect, from the base planner and the released adapters.

    python prepare_checkpoints.py --adapters checkpoints/adapters

Inputs
  third_party/Diffusion-Planner/best_model.pth   base planner (EMA weights are used)
  config/args.json                                model configuration
  --adapters                                      the unzipped release asset RouteOrAdapt_adapters_v1.zip
Outputs (paths used by the scripts in this folder)
  checkpoints/molora_v2_output/molora_full.pth            adapter stack, shared + 5 experts
  checkpoints/molora_v2_output/merged_<odd>.pth           shared + one expert merged into the base weights
  checkpoints/molora_v2_output/merged_shared_only.pth     shared module only, merged
  checkpoints/overnight_20260513/stage24_v2_joint_replay.pth, stage32_joint_replay_from_scratch_r4.pth,
  checkpoints/overnight_20260513/stage12_shared_park.pth
  outputs/rebuttal/e12_<run>_lora.pth                     Table 3 runs
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'third_party/Diffusion-Planner/diffusion_planner'))
sys.path.insert(0, str(HERE / 'third_party/Diffusion-Planner'))
sys.path.insert(0, str(HERE))
from diffusion_planner.model.diffusion_planner import Diffusion_Planner  # noqa: E402
from diffusion_planner.utils.config import Config  # noqa: E402
from train_molora import ODD_MAPS, N_EXPERTS, apply_molora, merge_molora_into_dit  # noqa: E402

SHARED_RANK, EXPERT_RANK, ALPHA = 4, 8, 32.0


def base_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('ema_state_dict') or ckpt['model']
    return {k.replace('module.', ''): v for k, v in sd.items()}


def stack(cfg, base, adapters, expert_merge=None):
    """Base planner with the adapter stack applied; optionally merge shared + one expert into the weights."""
    model = Diffusion_Planner(cfg).to('cpu')
    model.load_state_dict(base, strict=False)
    apply_molora(model.decoder.dit, n_experts=N_EXPERTS, shared_rank=SHARED_RANK, expert_rank=EXPERT_RANK, alpha=ALPHA)
    missing = set(adapters) - set(model.state_dict())
    assert not missing, f'unexpected adapter keys: {sorted(missing)[:3]}'
    model.load_state_dict(adapters, strict=False)
    if expert_merge is not False:
        merge_molora_into_dit(model.decoder.dit, expert_idx=expert_merge)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapters', type=Path, required=True)
    ap.add_argument('--base', type=Path, default=HERE / 'third_party/Diffusion-Planner/best_model.pth')
    ap.add_argument('--args', type=Path, default=HERE / 'config/args.json')
    ap.add_argument('--root', type=Path, default=HERE)
    a = ap.parse_args()
    cfg, base = Config(str(a.args)), base_state(a.base)
    mo = a.root / 'checkpoints/molora_v2_output'
    ov = a.root / 'checkpoints/overnight_20260513'
    rb = a.root / 'outputs/rebuttal'
    for d in (mo, ov, rb):
        d.mkdir(parents=True, exist_ok=True)

    sub = torch.load(a.adapters / 'substrate/shared_plus_experts.pth', map_location='cpu')
    torch.save({'model': stack(cfg, base, sub, expert_merge=False).state_dict()}, mo / 'molora_full.pth')
    for idx, name in enumerate(ODD_MAPS):
        torch.save({'model': stack(cfg, base, sub, expert_merge=idx).state_dict()}, mo / f'merged_{name}.pth')
    torch.save({'model': stack(cfg, base, sub, expert_merge=None).state_dict()}, mo / 'merged_shared_only.pth')

    for rel, out in [('joint_replay/pool_a_checkpoint.pth', 'stage24_v2_joint_replay.pth'),
                     ('joint_replay/from_scratch_r4.pth', 'stage32_joint_replay_from_scratch_r4.pth')]:
        ad = torch.load(a.adapters / rel, map_location='cpu')
        torch.save({'model': stack(cfg, base, ad, expert_merge=False).state_dict()}, ov / out)
    shutil.copy(a.adapters / 'shared_adapter_repair/shared_park.pth', ov / 'stage12_shared_park.pth')
    for p in glob.glob(str(a.adapters / 'matched_baselines/*.pth')):
        shutil.copy(p, rb / f'e12_{Path(p).stem}_lora.pth')
    print('checkpoints written under', a.root)


if __name__ == '__main__':
    main()
