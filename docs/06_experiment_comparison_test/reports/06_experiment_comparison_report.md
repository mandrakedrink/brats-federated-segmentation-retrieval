# 06 Experiment Comparison Report

Generated: 2026-09-15T03:36:37
Project root: `/home/mandrakedrink/projects/research_ts_bs`

## Experiments

- **Centralized SegResNet32 final CE-weighted — TEST** (`centralized_s32_cew`): loaded
  - metrics: `/home/mandrakedrink/projects/research_ts_bs/models/centralized_segresnet32_final/full_volume_inference_test/full_volume_test_metrics.csv`
  - history: `/home/mandrakedrink/projects/research_ts_bs/models/centralized_segresnet32_final/segresnet_final_history.csv`
- **FedAvg SegResNet32 final CE-weighted — TEST** (`fedavg_s32_cew`): loaded
  - metrics: `/home/mandrakedrink/projects/research_ts_bs/models/federated_fedavg_segresnet32_final/full_volume_inference_test/full_volume_test_metrics.csv`
  - history: `/home/mandrakedrink/projects/research_ts_bs/models/federated_fedavg_segresnet32_final/fedavg_history.csv`
- **FedProx SegResNet32 final CE-weighted — TEST** (`fedprox_s32_cew`): loaded
  - metrics: `/home/mandrakedrink/projects/research_ts_bs/models/federated_fedprox_segresnet32_final/full_volume_inference_test/full_volume_test_metrics.csv`

## Overall full-volume Dice

| experiment_key      | experiment_name                                  | experiment_short_name   | experiment_family   | experiment_stage    |   WT Dice |   TC Dice |   ET Dice |   Mean Dice |
|:--------------------|:-------------------------------------------------|:------------------------|:--------------------|:--------------------|----------:|----------:|----------:|------------:|
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST | C-S32 CEw               | centralized         | final_backbone_test |    0.9223 |    0.9202 |    0.874  |      0.9055 |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      | FedAvg-S32 CEw          | federated           | final_fedavg_test   |    0.9127 |    0.9051 |    0.8719 |      0.8966 |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     | FedProx-S32 CEw         | federated           | final_fedprox_test  |    0.9091 |    0.902  |    0.8628 |      0.8913 |

## Domain-stratified Dice

| experiment_key      | experiment_name                                  | experiment_short_name   | client_domain   |   n |   WT Dice |   TC Dice |   ET Dice |   Mean Dice |
|:--------------------|:-------------------------------------------------|:------------------------|:----------------|----:|----------:|----------:|----------:|------------:|
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST | C-S32 CEw               | TCGA-GBM        |  20 |    0.9189 |    0.888  |    0.8444 |      0.8838 |
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST | C-S32 CEw               | TCGA-LGG        |  13 |    0.9132 |    0.7511 |    0.5719 |      0.7454 |
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST | C-S32 CEw               | UCSF-PDGM       |  76 |    0.9199 |    0.9265 |    0.8967 |      0.9144 |
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST | C-S32 CEw               | UPENN-GBM       | 102 |    0.9259 |    0.9435 |    0.9014 |      0.9236 |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      | FedAvg-S32 CEw          | TCGA-GBM        |  20 |    0.9024 |    0.9067 |    0.8508 |      0.8866 |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      | FedAvg-S32 CEw          | TCGA-LGG        |  13 |    0.9131 |    0.4894 |    0.5412 |      0.6479 |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      | FedAvg-S32 CEw          | UCSF-PDGM       |  76 |    0.9118 |    0.9292 |    0.9005 |      0.9138 |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      | FedAvg-S32 CEw          | UPENN-GBM       | 102 |    0.9154 |    0.9397 |    0.8969 |      0.9174 |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     | FedProx-S32 CEw         | TCGA-GBM        |  20 |    0.8926 |    0.885  |    0.8333 |      0.8703 |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     | FedProx-S32 CEw         | TCGA-LGG        |  13 |    0.9133 |    0.5573 |    0.5211 |      0.6639 |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     | FedProx-S32 CEw         | UCSF-PDGM       |  76 |    0.9079 |    0.9213 |    0.8926 |      0.9073 |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     | FedProx-S32 CEw         | UPENN-GBM       | 102 |    0.9127 |    0.9349 |    0.8899 |      0.9125 |

## Pairwise deltas

| reference_key       | current_key    | metric    |   n |   mean_delta_current_minus_reference |   median_delta |   std_delta |   ci95_low |   ci95_high |   n_current_better |   n_current_equal |   n_current_worse |   wilcoxon_pvalue_delta_vs_zero |
|:--------------------|:---------------|:----------|----:|-------------------------------------:|---------------:|------------:|-----------:|------------:|-------------------:|------------------:|------------------:|--------------------------------:|
| centralized_s32_cew | fedavg_s32_cew | dice_WT   | 211 |                            -0.00958  |      -0.004904 |    0.04159  |  -0.015978 |   -0.004601 |                 68 |                 0 |               143 |                        0        |
| centralized_s32_cew | fedavg_s32_cew | dice_TC   | 211 |                            -0.015183 |      -0.001315 |    0.105836 |  -0.030604 |   -0.002385 |                 78 |                 0 |               133 |                        0.000105 |
| centralized_s32_cew | fedavg_s32_cew | dice_ET   | 211 |                            -0.002096 |      -0.002719 |    0.035903 |  -0.006937 |    0.002778 |                 65 |                 1 |               145 |                        3e-06    |
| centralized_s32_cew | fedavg_s32_cew | dice_mean | 211 |                            -0.008953 |      -0.003616 |    0.041709 |  -0.014959 |   -0.003625 |                 46 |                 0 |               165 |                        0        |

## Domain deltas

| client_domain   |   n | reference_key       | current_key    |   dice_WT_mean_delta |   dice_WT_ci95_low |   dice_WT_ci95_high |   dice_WT_median_delta |   dice_TC_mean_delta |   dice_TC_ci95_low |   dice_TC_ci95_high |   dice_TC_median_delta |   dice_ET_mean_delta |   dice_ET_ci95_low |   dice_ET_ci95_high |   dice_ET_median_delta |   dice_mean_mean_delta |   dice_mean_ci95_low |   dice_mean_ci95_high |   dice_mean_median_delta |
|:----------------|----:|:--------------------|:---------------|---------------------:|-------------------:|--------------------:|-----------------------:|---------------------:|-------------------:|--------------------:|-----------------------:|---------------------:|-------------------:|--------------------:|-----------------------:|-----------------------:|---------------------:|----------------------:|-------------------------:|
| TCGA-GBM        |  20 | centralized_s32_cew | fedavg_s32_cew |            -0.016503 |          -0.031811 |           -0.004467 |              -0.003496 |             0.018687 |          -0.004609 |            0.055022 |              -0.00078  |             0.006444 |          -0.00941  |            0.031128 |              -0.003768 |               0.002876 |            -0.01103  |              0.023487 |                -0.003007 |
| TCGA-LGG        |  13 | centralized_s32_cew | fedavg_s32_cew |            -0.000142 |          -0.008345 |            0.007962 |              -0.002313 |            -0.261696 |          -0.435423 |           -0.110836 |              -0.153611 |            -0.03071  |          -0.074087 |            0.006829 |              -0.000236 |              -0.097516 |            -0.154403 |             -0.04623  |                -0.081552 |
| UCSF-PDGM       |  76 | centralized_s32_cew | fedavg_s32_cew |            -0.008105 |          -0.023397 |            0.001739 |              -0.001574 |             0.00276  |          -0.001694 |            0.008376 |              -0.000928 |             0.003758 |          -0.002249 |            0.011119 |              -0.002327 |              -0.000529 |            -0.004992 |              0.004635 |                -0.002614 |
| UPENN-GBM       | 102 | centralized_s32_cew | fedavg_s32_cew |            -0.010525 |          -0.016394 |           -0.004754 |              -0.007993 |            -0.003775 |          -0.012347 |            0.002081 |              -0.001235 |            -0.004485 |          -0.0099   |            0.00025  |              -0.002997 |              -0.006262 |            -0.011147 |             -0.002125 |                -0.003998 |

## Failure counts

| experiment_key      | experiment_name                                  | experiment_short_name   |   n_cases |   n_gt_ET_present |   n_gt_ET_absent |   n_pred_ET_present |   n_ET_false_positive |   n_ET_false_negative_strict |   n_ET_near_miss_low_pred_ratio |   n_ET_overseg_ratio_flag |   n_ET_low_dice |   n_TC_low_dice |   n_WT_low_dice |
|:--------------------|:-------------------------------------------------|:------------------------|----------:|------------------:|-----------------:|--------------------:|----------------------:|-----------------------------:|--------------------------------:|--------------------------:|----------------:|----------------:|----------------:|
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST | C-S32 CEw               |       211 |               208 |                3 |                 210 |                     2 |                            0 |                               2 |                         1 |               6 |               2 |               0 |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      | FedAvg-S32 CEw          |       211 |               208 |                3 |                 210 |                     2 |                            0 |                               2 |                         1 |               7 |               8 |               1 |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     | FedProx-S32 CEw         |       211 |               208 |                3 |                 209 |                     2 |                            1 |                               2 |                         3 |               8 |               8 |               1 |

## TCGA-LGG analysis

| patient_id      | client_domain   | split   |   dice_WT_centralized_s32_cew |   dice_TC_centralized_s32_cew |   dice_ET_centralized_s32_cew |   dice_WT_fedavg_s32_cew |   dice_TC_fedavg_s32_cew |   dice_ET_fedavg_s32_cew |   dice_WT_delta |   dice_TC_delta |   dice_ET_delta |   dice_mean_delta |   gt_WT_vox_centralized_s32_cew |   gt_TC_vox_centralized_s32_cew |   gt_ET_vox_centralized_s32_cew |   pred_WT_vox_centralized_s32_cew |   pred_TC_vox_centralized_s32_cew |   pred_ET_vox_centralized_s32_cew |   pred_WT_vox_fedavg_s32_cew |   pred_TC_vox_fedavg_s32_cew |   pred_ET_vox_fedavg_s32_cew |   pred_to_gt_ET_ratio_centralized_s32_cew |   pred_to_gt_ET_ratio_fedavg_s32_cew | gt_ET_bin_centralized_s32_cew   |
|:----------------|:----------------|:--------|------------------------------:|------------------------------:|------------------------------:|-------------------------:|-------------------------:|-------------------------:|----------------:|----------------:|----------------:|------------------:|--------------------------------:|--------------------------------:|--------------------------------:|----------------------------------:|----------------------------------:|----------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|------------------------------------------:|-------------------------------------:|:--------------------------------|
| BraTS2021_01527 | TCGA-LGG        | test    |                        0.924  |                        0.9286 |                        0.285  |                   0.9442 |                   0.775  |                   0.095  |          0.0202 |         -0.1536 |         -0.1901 |           -0.1078 |                           87527 |                           62120 |                             119 |                            101618 |                             57658 |                               295 |                        96814 |                        39495 |                          239 |                                    2.479  |                               2.0084 | ET_tiny                         |
| BraTS2021_01531 | TCGA-LGG        | test    |                        0.9218 |                        0.5357 |                        0.3444 |                   0.9281 |                   0.4742 |                   0.1548 |          0.0063 |         -0.0614 |         -0.1896 |           -0.0816 |                          125353 |                          125353 |                            1194 |                            115137 |                             45912 |                              4910 |                       116768 |                        39837 |                        13676 |                                    4.1122 |                              11.4539 | ET_medium                       |
| BraTS2021_01505 | TCGA-LGG        | test    |                        0.9057 |                        0.7868 |                        0.8968 |                   0.9155 |                   0.0294 |                   0.8303 |          0.0099 |         -0.7574 |         -0.0665 |           -0.2714 |                           95449 |                           67018 |                            1072 |                            105660 |                             50493 |                              1118 |                       102358 |                         1213 |                         1108 |                                    1.0429 |                               1.0336 | ET_medium                       |
| BraTS2021_01522 | TCGA-LGG        | test    |                        0.9449 |                        0.7745 |                        0.8915 |                   0.9412 |                   0.7537 |                   0.8594 |         -0.0037 |         -0.0208 |         -0.0321 |           -0.0189 |                           20725 |                            6499 |                            3794 |                             21445 |                              4448 |                              3912 |                        21818 |                         4299 |                         3879 |                                    1.0311 |                               1.0224 | ET_medium                       |
| BraTS2021_01481 | TCGA-LGG        | test    |                        0.9083 |                        0.8816 |                        0.7694 |                   0.8983 |                   0.8745 |                   0.7501 |         -0.01   |         -0.0071 |         -0.0194 |           -0.0122 |                           56512 |                            4076 |                            2575 |                             62554 |                              4331 |                              4113 |                        60584 |                         4434 |                         4291 |                                    1.5973 |                               1.6664 | ET_medium                       |
| BraTS2021_01493 | TCGA-LGG        | test    |                        0.9206 |                        0.7847 |                        0.8913 |                   0.9111 |                   0.753  |                   0.8854 |         -0.0095 |         -0.0316 |         -0.0058 |           -0.0157 |                           77001 |                           37156 |                           18283 |                             86441 |                             39423 |                             17389 |                        86779 |                        34817 |                        18598 |                                    0.9511 |                               1.0172 | ET_large                        |
| BraTS2021_01536 | TCGA-LGG        | test    |                        0.7745 |                        0.5914 |                        0.8134 |                   0.8011 |                   0.2502 |                   0.8131 |          0.0266 |         -0.3412 |         -0.0002 |           -0.1049 |                           49858 |                           10778 |                            2093 |                             78883 |                              5116 |                              1647 |                        74582 |                         1578 |                         1562 |                                    0.7869 |                               0.7463 | ET_medium                       |
| BraTS2021_01477 | TCGA-LGG        | test    |                        0.9589 |                        0.7565 |                        0      |                   0.9472 |                   0.0204 |                   0      |         -0.0117 |         -0.7362 |         -0      |           -0.2493 |                          195433 |                           77985 |                               0 |                            204655 |                            110496 |                               108 |                       211161 |                         1121 |                          674 |                                  nan      |                             nan      | ET_absent                       |
| BraTS2021_01491 | TCGA-LGG        | test    |                        0.9076 |                        0.6374 |                        0      |                   0.8918 |                   0.2203 |                   0      |         -0.0158 |         -0.4172 |         -0      |           -0.1443 |                           23844 |                           21004 |                               0 |                             22087 |                             10001 |                               254 |                        20292 |                         2603 |                          969 |                                  nan      |                             nan      | ET_absent                       |
| BraTS2021_01526 | TCGA-LGG        | test    |                        0.9506 |                        0.7637 |                        1      |                   0.9564 |                   0      |                   1      |          0.0058 |         -0.7637 |          0      |           -0.2526 |                           65338 |                           28605 |                               0 |                             66649 |                             45920 |                                 0 |                        63727 |                            0 |                            0 |                                  nan      |                             nan      | ET_absent                       |
| BraTS2021_01528 | TCGA-LGG        | test    |                        0.9208 |                        0.8644 |                        0      |                   0.9185 |                   0.786  |                   0.0004 |         -0.0023 |         -0.0784 |          0.0004 |           -0.0268 |                          232098 |                          104710 |                            4674 |                            249195 |                            126874 |                               361 |                       237425 |                        81963 |                          934 |                                    0.0772 |                               0.1998 | ET_medium                       |
| BraTS2021_01495 | TCGA-LGG        | test    |                        0.9074 |                        0.6702 |                        0.8678 |                   0.8777 |                   0.8433 |                   0.9107 |         -0.0297 |          0.1731 |          0.0429 |            0.0621 |                          220284 |                            6321 |                            4085 |                            222140 |                              8624 |                              4781 |                       224349 |                         4660 |                         4660 |                                    1.1704 |                               1.1408 | ET_medium                       |
| BraTS2021_01513 | TCGA-LGG        | test    |                        0.9267 |                        0.7889 |                        0.6748 |                   0.9388 |                   0.5824 |                   0.7361 |          0.0121 |         -0.2065 |          0.0612 |           -0.0444 |                          202427 |                          153111 |                            7070 |                            208701 |                            148061 |                              3872 |                       195462 |                        69209 |                         9075 |                                    0.5477 |                               1.2836 | ET_large                        |

## Generated outputs

### Tables

- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/00_manifest_split_domain_counts.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/00_manifest_test_domain_counts.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/01_experiment_registry.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/02_all_full_volume_metrics_long.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/03_integrity_checks.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/04_patient_set_alignment.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/05_overall_full_volume_performance_paper.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/06_overall_full_volume_performance_stats.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/07_domain_full_volume_performance_paper.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/08_domain_full_volume_performance_stats.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/09_et_present_absent_performance_paper.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/10_et_present_absent_performance_stats.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/11_et_volume_bin_performance_paper.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/12_et_volume_bin_performance_stats.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/13_domain_by_et_present_performance_paper.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/14_domain_by_et_present_performance_stats.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/15_failure_counts_overall.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/16_failure_counts_by_domain.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/17_volume_error_summary.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/18_pairwise_centralized_s32_cew_vs_fedavg_s32_cew_case_level.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/19_pairwise_centralized_s32_cew_vs_fedavg_s32_cew_delta_summary.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/20_pairwise_centralized_s32_cew_vs_fedavg_s32_cew_domain_delta_summary.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/21_tcga_lgg_deep_dive_centralized_s32_cew_vs_fedavg_s32_cew.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/22_tcga_lgg_all_experiments_case_rows.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_centralized_s32_cew_dice_ET.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_centralized_s32_cew_dice_TC.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_centralized_s32_cew_dice_WT.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_fedavg_s32_cew_dice_ET.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_fedavg_s32_cew_dice_TC.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_fedavg_s32_cew_dice_WT.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_fedprox_s32_cew_dice_ET.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_fedprox_s32_cew_dice_TC.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/23_worst_fedprox_s32_cew_dice_WT.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/24_worst_negative_delta_centralized_s32_cew_vs_fedavg_s32_cew_dice_ET.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/24_worst_negative_delta_centralized_s32_cew_vs_fedavg_s32_cew_dice_TC.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/24_worst_negative_delta_centralized_s32_cew_vs_fedavg_s32_cew_dice_WT.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/25_training_history_summary.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/26_training_histories_combined.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/27_output_checks.csv`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/tables/27_readiness_gate.csv`

### Figures

- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/01_overall_dice_by_experiment.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/02_domain_et_dice.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/02_domain_mean_dice.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/02_domain_tc_dice.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/02_domain_wt_dice.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/03_primary_domain_delta_heatmap.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/04_case_delta_distribution_ET.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/04_case_delta_distribution_TC.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/04_case_delta_distribution_WT.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/04_case_delta_distribution_mean.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/05_tcga_lgg_case_ET.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/05_tcga_lgg_case_TC.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/05_tcga_lgg_case_WT.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/06_fedavg_validation_dice_curves.png`
- `/home/mandrakedrink/projects/research_ts_bs/docs/06_experiment_comparison_test/figures/07_fedavg_loss_curves.png`
