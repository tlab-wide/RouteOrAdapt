# Evaluation code

These are the scripts that produced the files in `../results/`. They are released as run, with local paths replaced by the relative folders below. Running them needs the held-out maps and the base planner checkpoint, which are not in this repository.

## Setup

```bash
cd pipeline
git clone https://github.com/tier4/Diffusion-Planner third_party/Diffusion-Planner
git -C third_party/Diffusion-Planner checkout 8253a4f
git -C third_party/Diffusion-Planner apply ../../patches/tier4_utils_load_npz_dtype.patch   # dtype fix used for training data
cp config/args.json third_party/Diffusion-Planner/args.json
# put the base planner checkpoint at third_party/Diffusion-Planner/best_model.pth
# download RouteOrAdapt_adapters_v1.zip from the GitHub release and unzip it to checkpoints/adapters
python prepare_checkpoints.py --adapters checkpoints/adapters
```

Map conversion uses the Python bindings of the Autoware lanelet2 extension (`/opt/autoware`). The lanelet2 maps of the held-out regions are proprietary and not released; the scripts expect them under `maps/unseen_extracted/`.

## Layout used by the scripts

| Folder | Content |
|---|---|
| `third_party/Diffusion-Planner/` | the PyTorch planner code, the base checkpoint and `args.json` |
| `checkpoints/` | adapter checkpoints rebuilt by `prepare_checkpoints.py`, and training data lists |
| `maps/unseen_extracted/` | lanelet2 maps of the three held-out regions (not released) |
| `outputs/` | where the scripts write their JSON results |

`train_molora.py` implements the shared-plus-expert adapter stack, and `train_reward_backprop.py` the differentiable DPM-Solver++ sampler that every script uses for rollouts.

## Script to result map

| Script | Result file in `../results/` | Paper |
|---|---|---|
| `overnight_stage1_multiego.py` | `routing_pool/per_expert_rollouts.json`; also defines the scene sampler of the routing pool | Sec. 4 |
| `overnight_stage3_scr.py`, `overnight_stage5_scr_eval.py` | `routing_pool/scene_mlp_and_self_consistency.json` | Sec. 4 |
| `overnight_stage8_lfs.py` | `routing_pool/lane_following.json` | Sec. 4 |
| `overnight_stage9_dlr.py`, `overnight_stage9b_dlr_kendall.py` | `routing_pool/denoising_loss.json` | Sec. 4 |
| `overnight_stage10_gps_mix.py` | `routing_pool/vmf_mixture.json` | Sec. 5.1 |
| `overnight_stage11_conformal.py`, `overnight_stage11_conformal_v3.py` | `routing_pool/conformal_corridor.json` | Sec. 4 |
| `reward_based_selector.py` | `routing_pool/reward_as_scorer.json` | Sec. 4 |
| `forensics.py`, `tent_init_variants.py`, `tent_fullblood.py`, `hypernet_ln.py` | `routing_pool/variance_collapse_forensics.json` | Sec. 5.4 |
| `rebuttal_e4_paired_tests.py` | `routing_pool/paired_tests.json` | Sec. 4 |
| `rebuttal_e3_loocv_router.py` | `routing_pool/loocv_handcrafted_features.json` | Sec. 4 |
| `rebuttal_e9_loocv_encoder.py`, `encoder_feature_mi_conformal.py` | `routing_pool/loocv_encoder_embedding.json` | Sec. 4 |
| `rebuttal_e18_expanded_pool.py` | `expanded_pool_104/expanded_pool.json` | Sec. 4 |
| `overnight_stage12_park_repair.py` | `shared_adapter_repair/` | Sec. 5.2 |
| `aichi_disjoint_split.py` | the train/eval splits in `../splits/` | Sec. 3 |
| `overnight_stage24_v2_joint_replay.py`, `jointreplay_aichi_safety.py` | `weight_space/stage24_v2_joint_replay.json`, `routing_pool/joint_replay_and_safety_proxies.json` | Sec. 4, 5.3 |
| `rebuttal_e12_matched_baselines.py` | `matched_baselines/runs/` | Table 3 |
| `rebuttal_e12_aggregate.py` | earlier three-seed aggregate; superseded by `../analysis/reproduce_paper_numbers.py` | |
| `rebuttal_e10_prod_perscene.py` | `matched_baselines/source_forgetting_per_scene.json` | Sec. 5.3 |
| `rebuttal_e6_oracle_slice.py` | `matched_baselines/oracle_per_seed_slice.json` | Sec. 4 |
| `rebuttal_e13_safety_baselines.py` | `matched_baselines/safety_proxies.json` | Sec. 5.3 |
| `overnight_stage26_sample_complexity.py` | `sample_complexity/`, `weight_space/stage26_sample_complexity.json` | Fig. 5 |
| `overnight_stage32_rank_sweep.py` | `weight_space/stage32_rank_sweep_run3.json` | Appendix |
| `rebuttal_e11b_kashiwacity_full.py` | `switching_regions/suburb_adjacent_to_kashiwa.json` | Sec. 7 |
| `rebuttal_e8_shinjuku_switching.py` | `switching_regions/dense_urban_nearest_hongo.json` | Sec. 7 |

The script names follow the order in which the experiments were run (`overnight_stage*`) or the review round in which they were added (`rebuttal_e*`).
