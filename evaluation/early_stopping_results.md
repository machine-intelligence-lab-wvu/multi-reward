# Early-stopping baseline analysis (T0.1) — regenerated 2026-08-31 from eval_results_final on $PROJECT

Inputs: single_multi_reward_ablation/ wandb exports (length+entropy) + SUMMARY.csv, extended past ckpt 280 by trajectory_postfix_1.5b/SUMMARY.csv. Rebuild with evaluation/prepare_early_stopping_inputs.py.

## No burn-in (rules armed from step 0)

## intuitor  (rules armed after step 0)
oracle peak : step 40: GSM8K 72.9, M500 49.8   (label-based -- not available in RLIF)
end of run  : step 340: GSM8K 0.0, M500 0.0

| rule | tau | trigger step | selected ckpt & accuracy |
|---|---|---|---|
| len_drop | 0.15 | 21.0 | step 20: GSM8K 13.1, M500 30.0 |
| len_rise | 0.15 | 234.0 | step 220: GSM8K 0.0, M500 0.0 |
| two_sided | 0.15 | 21.0 | step 20: GSM8K 13.1, M500 30.0 |
| len_drop | 0.25 | 21.0 | step 20: GSM8K 13.1, M500 30.0 |
| len_rise | 0.25 | 234.0 | step 220: GSM8K 0.0, M500 0.0 |
| two_sided | 0.25 | 21.0 | step 20: GSM8K 13.1, M500 30.0 |
| len_drop | 0.4 | 28.0 | step 20: GSM8K 13.1, M500 30.0 |
| len_rise | 0.4 | 234.0 | step 220: GSM8K 0.0, M500 0.0 |
| two_sided | 0.4 | 28.0 | step 20: GSM8K 13.1, M500 30.0 |
| ent_floor | <0.5 nats | 29.0 | step 20: GSM8K 13.1, M500 30.0 |

## cluster_only  (rules armed after step 0)
oracle peak : step 60: GSM8K 75.0, M500 52.6   (label-based -- not available in RLIF)
end of run  : step 280: GSM8K 0.0, M500 1.4

| rule | tau | trigger step | selected ckpt & accuracy |
|---|---|---|---|
| len_drop | 0.15 | 23.0 | step 20: GSM8K 13.6, M500 27.2 |
| len_rise | 0.15 | 190.0 | step 180: GSM8K 0.0, M500 1.4 |
| two_sided | 0.15 | 23.0 | step 20: GSM8K 13.6, M500 27.2 |
| len_drop | 0.25 | 27.0 | step 20: GSM8K 13.6, M500 27.2 |
| len_rise | 0.25 | never | step 280: GSM8K 0.0, M500 1.4 |
| two_sided | 0.25 | 27.0 | step 20: GSM8K 13.6, M500 27.2 |
| len_drop | 0.4 | never | step 280: GSM8K 0.0, M500 1.4 |
| len_rise | 0.4 | never | step 280: GSM8K 0.0, M500 1.4 |
| two_sided | 0.4 | never | step 280: GSM8K 0.0, M500 1.4 |
| ent_floor | <0.5 nats | 34.0 | step 20: GSM8K 13.6, M500 27.2 |

## multi_reward  (rules armed after step 0)
oracle peak : step 40: GSM8K 73.2, M500 53.6   (label-based -- not available in RLIF)
end of run  : step 340: GSM8K 1.1, M500 2.4

| rule | tau | trigger step | selected ckpt & accuracy |
|---|---|---|---|
| len_drop | 0.15 | 18.0 | step 20: GSM8K 15.5, M500 29.8 |
| len_rise | 0.15 | 53.0 | step 40: GSM8K 73.2, M500 53.6 |
| two_sided | 0.15 | 18.0 | step 20: GSM8K 15.5, M500 29.8 |
| len_drop | 0.25 | 22.0 | step 20: GSM8K 15.5, M500 29.8 |
| len_rise | 0.25 | never | step 340: GSM8K 1.1, M500 2.4 |
| two_sided | 0.25 | 22.0 | step 20: GSM8K 15.5, M500 29.8 |
| len_drop | 0.4 | 27.0 | step 20: GSM8K 15.5, M500 29.8 |
| len_rise | 0.4 | never | step 340: GSM8K 1.1, M500 2.4 |
| two_sided | 0.4 | 27.0 | step 20: GSM8K 15.5, M500 29.8 |
| ent_floor | <0.5 nats | 29.0 | step 20: GSM8K 15.5, M500 29.8 |

## Burn-in 60 (rules armed after the format-learning transient)

## intuitor  (rules armed after step 60)
oracle peak : step 40: GSM8K 72.9, M500 49.8   (label-based -- not available in RLIF)
end of run  : step 340: GSM8K 0.0, M500 0.0

| rule | tau | trigger step | selected ckpt & accuracy |
|---|---|---|---|
| len_drop | 0.15 | 158.0 | step 140: GSM8K 45.0, M500 32.4 |
| len_rise | 0.15 | 234.0 | step 220: GSM8K 0.0, M500 0.0 |
| two_sided | 0.15 | 158.0 | step 140: GSM8K 45.0, M500 32.4 |
| len_drop | 0.25 | 191.0 | step 180: GSM8K 3.9, M500 11.2 |
| len_rise | 0.25 | 234.0 | step 220: GSM8K 0.0, M500 0.0 |
| two_sided | 0.25 | 191.0 | step 180: GSM8K 3.9, M500 11.2 |
| len_drop | 0.4 | 192.0 | step 180: GSM8K 3.9, M500 11.2 |
| len_rise | 0.4 | 234.0 | step 220: GSM8K 0.0, M500 0.0 |
| two_sided | 0.4 | 192.0 | step 180: GSM8K 3.9, M500 11.2 |
| ent_floor | <0.5 nats | 60.0 | step 60: GSM8K 70.4, M500 51.4 |

## cluster_only  (rules armed after step 60)
oracle peak : step 60: GSM8K 75.0, M500 52.6   (label-based -- not available in RLIF)
end of run  : step 280: GSM8K 0.0, M500 1.4

| rule | tau | trigger step | selected ckpt & accuracy |
|---|---|---|---|
| len_drop | 0.15 | never | step 280: GSM8K 0.0, M500 1.4 |
| len_rise | 0.15 | 190.0 | step 180: GSM8K 0.0, M500 1.4 |
| two_sided | 0.15 | 190.0 | step 180: GSM8K 0.0, M500 1.4 |
| len_drop | 0.25 | never | step 280: GSM8K 0.0, M500 1.4 |
| len_rise | 0.25 | never | step 280: GSM8K 0.0, M500 1.4 |
| two_sided | 0.25 | never | step 280: GSM8K 0.0, M500 1.4 |
| len_drop | 0.4 | never | step 280: GSM8K 0.0, M500 1.4 |
| len_rise | 0.4 | never | step 280: GSM8K 0.0, M500 1.4 |
| two_sided | 0.4 | never | step 280: GSM8K 0.0, M500 1.4 |
| ent_floor | <0.5 nats | 60.0 | step 60: GSM8K 75.0, M500 52.6 |

## multi_reward  (rules armed after step 60)
oracle peak : step 40: GSM8K 73.2, M500 53.6   (label-based -- not available in RLIF)
end of run  : step 340: GSM8K 1.1, M500 2.4

| rule | tau | trigger step | selected ckpt & accuracy |
|---|---|---|---|
| len_drop | 0.15 | 288.0 | step 280: GSM8K 1.1, M500 2.4 |
| len_rise | 0.15 | never | step 340: GSM8K 1.1, M500 2.4 |
| two_sided | 0.15 | 288.0 | step 280: GSM8K 1.1, M500 2.4 |
| len_drop | 0.25 | 346.0 | step 340: GSM8K 1.1, M500 2.4 |
| len_rise | 0.25 | never | step 340: GSM8K 1.1, M500 2.4 |
| two_sided | 0.25 | 346.0 | step 340: GSM8K 1.1, M500 2.4 |
| len_drop | 0.4 | never | step 340: GSM8K 1.1, M500 2.4 |
| len_rise | 0.4 | never | step 340: GSM8K 1.1, M500 2.4 |
| two_sided | 0.4 | never | step 340: GSM8K 1.1, M500 2.4 |
| ent_floor | <0.5 nats | 60.0 | step 60: GSM8K 71.7, M500 52.8 |
