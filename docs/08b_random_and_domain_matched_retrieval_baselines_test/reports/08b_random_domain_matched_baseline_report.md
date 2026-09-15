# 08b Random and Domain-Matched Retrieval Baseline Report

## Evaluation setup

- Retrieval space: `embedding`
- k: `5`
- Random repeats: `1000`

## Baselines

- `global_random`: random non-self neighbors from the full evaluation set.
- `same_domain_random_strict`: random neighbors restricted to the query domain.
- `domain_distribution_matched_random`: random neighbors matched to the retrieved-domain composition of the learned retrieval result.

## Domain-matched control summary

| Experiment          | Verdict                                                          |   Distance metrics better |   Distance metrics total |   Consistency metrics better |   Consistency metrics total |
|:--------------------|:-----------------------------------------------------------------|--------------------------:|-------------------------:|-----------------------------:|----------------------------:|
| centralized_s32_cew | stronger_than_domain_matched_random_on_multiple_distance_metrics |                         4 |                        4 |                            0 |                           1 |
| fedavg_s32_cew      | stronger_than_domain_matched_random_on_multiple_distance_metrics |                         4 |                        4 |                            0 |                           1 |
| fedprox_s32_cew     | stronger_than_domain_matched_random_on_multiple_distance_metrics |                         4 |                        4 |                            0 |                           1 |

## Primary embedding comparison at k=5

| Experiment          | Metric                        |   Observed |   Global random |   Domain-matched random | DM flag                              |
|:--------------------|:------------------------------|-----------:|----------------:|------------------------:|:-------------------------------------|
| centralized_s32_cew | mean_abs_log1p_ET_diff        |     0.7504 |          1.4886 |                  1.3279 | better_than_random                   |
| centralized_s32_cew | mean_abs_log1p_TC_diff        |     0.6311 |          1.2702 |                  1.2642 | better_than_random                   |
| centralized_s32_cew | mean_abs_log1p_WT_diff        |     0.5685 |          0.8634 |                  0.8548 | better_than_random                   |
| centralized_s32_cew | mean_tumor_feature_z_distance |     1.1972 |          1.3878 |                  1.3744 | better_than_random                   |
| centralized_s32_cew | same_ET_presence_rate         |     0.9848 |          0.9716 |                  0.9803 | within_random_interval               |
| centralized_s32_cew | same_domain_rate              |     0.6796 |          0.3736 |                  0.6796 | domain_leakage_or_composition_metric |
| fedavg_s32_cew      | mean_abs_log1p_ET_diff        |     0.6867 |          1.4886 |                  1.3194 | better_than_random                   |
| fedavg_s32_cew      | mean_abs_log1p_TC_diff        |     0.5774 |          1.2702 |                  1.2626 | better_than_random                   |
| fedavg_s32_cew      | mean_abs_log1p_WT_diff        |     0.4531 |          0.8634 |                  0.8482 | better_than_random                   |
| fedavg_s32_cew      | mean_tumor_feature_z_distance |     1.1566 |          1.3878 |                  1.375  | better_than_random                   |
| fedavg_s32_cew      | same_ET_presence_rate         |     0.9848 |          0.9716 |                  0.9803 | within_random_interval               |
| fedavg_s32_cew      | same_domain_rate              |     0.5697 |          0.3736 |                  0.5697 | domain_leakage_or_composition_metric |
| fedprox_s32_cew     | mean_abs_log1p_ET_diff        |     0.7533 |          1.4886 |                  1.3385 | better_than_random                   |
| fedprox_s32_cew     | mean_abs_log1p_TC_diff        |     0.5991 |          1.2702 |                  1.2646 | better_than_random                   |
| fedprox_s32_cew     | mean_abs_log1p_WT_diff        |     0.5096 |          0.8634 |                  0.8547 | better_than_random                   |
| fedprox_s32_cew     | mean_tumor_feature_z_distance |     1.1845 |          1.3878 |                  1.3766 | better_than_random                   |
| fedprox_s32_cew     | same_ET_presence_rate         |     0.981  |          0.9716 |                  0.9781 | within_random_interval               |
| fedprox_s32_cew     | same_domain_rate              |     0.5545 |          0.3736 |                  0.5545 | domain_leakage_or_composition_metric |

## Metric notes

- Lower WT, TC, ET volume differences and tumor-feature z-distance indicate closer tumor-phenotype similarity.
- Same-domain rate measures domain concentration and is not treated as a retrieval-quality metric by itself.
- The domain-distribution-matched baseline preserves retrieved-domain composition when testing phenotype-distance improvement.

## Outputs

- Primary comparison table: `/home/mandrakedrink/projects/research_ts_bs/docs/08b_random_and_domain_matched_retrieval_baselines_test/tables/09_primary_baseline_comparison_embedding_k5.csv`
- Tables: `/home/mandrakedrink/projects/research_ts_bs/docs/08b_random_and_domain_matched_retrieval_baselines_test/tables`
- Reports: `/home/mandrakedrink/projects/research_ts_bs/docs/08b_random_and_domain_matched_retrieval_baselines_test/reports`
