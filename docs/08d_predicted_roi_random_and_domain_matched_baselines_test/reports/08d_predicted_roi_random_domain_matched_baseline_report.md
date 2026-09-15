# 08d Predicted-mask ROI Baseline Report

## Evaluation setup

- Source 08c directory: `docs/08c_predicted_mask_roi_embedding_retrieval_test`
- Patients: `211`
- Split: `['test']`
- ROI source: predicted WT center
- GT ROI fallback: disabled
- Primary retrieval space: `embedding`
- k: `5`
- Random repeats: `1000`

## Baselines

- `global_random`: random non-self neighbors from the full test set.
- `same_domain_random_strict`: random neighbors restricted to the query domain.
- `domain_distribution_matched_random`: random neighbors sampled to reproduce the retrieved-domain composition of the learned result.

## Predicted-ROI localization

| Experiment          |   No-WT fallback rate |   Mean center distance |   Median center distance |
|:--------------------|----------------------:|-----------------------:|-------------------------:|
| centralized_s32_cew |                     0 |                 1.9365 |                   1.4142 |
| fedavg_s32_cew      |                     0 |                 2.4682 |                   1.4142 |
| fedprox_s32_cew     |                     0 |                 2.5301 |                   1.4142 |

## Domain-matched control summary

| Experiment          | Verdict                                                          |   Distance metrics better |   Distance metrics total |   Consistency metrics better |   Consistency metrics total |
|:--------------------|:-----------------------------------------------------------------|--------------------------:|-------------------------:|-----------------------------:|----------------------------:|
| centralized_s32_cew | stronger_than_domain_matched_random_on_multiple_distance_metrics |                         4 |                        4 |                            0 |                           1 |
| fedavg_s32_cew      | stronger_than_domain_matched_random_on_multiple_distance_metrics |                         4 |                        4 |                            1 |                           1 |
| fedprox_s32_cew     | stronger_than_domain_matched_random_on_multiple_distance_metrics |                         4 |                        4 |                            0 |                           1 |

## Primary embedding comparison at k=5

| Experiment          | Metric                        |   Observed |   Global random |   Domain-matched random | DM flag                              |
|:--------------------|:------------------------------|-----------:|----------------:|------------------------:|:-------------------------------------|
| centralized_s32_cew | mean_abs_log1p_ET_diff        |     0.7549 |          1.4898 |                  1.3253 | better_than_random                   |
| centralized_s32_cew | mean_abs_log1p_TC_diff        |     0.633  |          1.2715 |                  1.2655 | better_than_random                   |
| centralized_s32_cew | mean_abs_log1p_WT_diff        |     0.5715 |          0.8624 |                  0.8543 | better_than_random                   |
| centralized_s32_cew | mean_tumor_feature_z_distance |     1.2    |          1.3879 |                  1.374  | better_than_random                   |
| centralized_s32_cew | same_ET_presence_rate         |     0.9848 |          0.9717 |                  0.9804 | within_random_interval               |
| centralized_s32_cew | same_domain_rate              |     0.6711 |          0.3731 |                  0.6711 | domain_leakage_or_composition_metric |
| fedavg_s32_cew      | mean_abs_log1p_ET_diff        |     0.6925 |          1.4898 |                  1.3238 | better_than_random                   |
| fedavg_s32_cew      | mean_abs_log1p_TC_diff        |     0.587  |          1.2715 |                  1.2649 | better_than_random                   |
| fedavg_s32_cew      | mean_abs_log1p_WT_diff        |     0.4693 |          0.8624 |                  0.8517 | better_than_random                   |
| fedavg_s32_cew      | mean_tumor_feature_z_distance |     1.1561 |          1.3879 |                  1.375  | better_than_random                   |
| fedavg_s32_cew      | same_ET_presence_rate         |     0.9848 |          0.9717 |                  0.9787 | better_than_random                   |
| fedavg_s32_cew      | same_domain_rate              |     0.5678 |          0.3731 |                  0.5678 | domain_leakage_or_composition_metric |
| fedprox_s32_cew     | mean_abs_log1p_ET_diff        |     0.7608 |          1.4898 |                  1.3372 | better_than_random                   |
| fedprox_s32_cew     | mean_abs_log1p_TC_diff        |     0.5998 |          1.2715 |                  1.2615 | better_than_random                   |
| fedprox_s32_cew     | mean_abs_log1p_WT_diff        |     0.4996 |          0.8624 |                  0.8529 | better_than_random                   |
| fedprox_s32_cew     | mean_tumor_feature_z_distance |     1.1799 |          1.3879 |                  1.3778 | better_than_random                   |
| fedprox_s32_cew     | same_ET_presence_rate         |     0.981  |          0.9717 |                  0.9796 | within_random_interval               |
| fedprox_s32_cew     | same_domain_rate              |     0.5611 |          0.3731 |                  0.5611 | domain_leakage_or_composition_metric |

## Best observed phenotype distances

| Metric           | Experiment     |   Observed |
|:-----------------|:---------------|-----------:|
| WT volume diff   | fedavg_s32_cew |     0.4693 |
| TC volume diff   | fedavg_s32_cew |     0.587  |
| ET volume diff   | fedavg_s32_cew |     0.6925 |
| Tumor z-distance | fedavg_s32_cew |     1.1561 |

## Result summary

All enabled experiments outperform the domain-distribution-matched random control on all four primary phenotype-distance metrics.

## Method notes

- Lower WT, TC, ET volume differences and tumor z-distance indicate closer tumor-phenotype similarity.
- The domain-distribution-matched control preserves retrieved-domain composition and therefore controls for domain composition in the primary phenotype-distance comparison.
- Same-domain rate is reported as a domain-clustering diagnostic, not as a retrieval-quality metric.
- Ground-truth tumor features are used to score retrieved-neighbor phenotype similarity; they are not used for predicted-ROI localization or primary embedding ranking.

## Outputs

- Primary comparison: `/home/mandrakedrink/projects/research_ts_bs/docs/08d_predicted_roi_random_and_domain_matched_baselines_test/tables/09_primary_baseline_comparison_embedding_k5.csv`
- Domain-matched control: `/home/mandrakedrink/projects/research_ts_bs/docs/08d_predicted_roi_random_and_domain_matched_baselines_test/tables/10_domain_distribution_matched_control_embedding_k5.csv`
- Verdict table: `/home/mandrakedrink/projects/research_ts_bs/docs/08d_predicted_roi_random_and_domain_matched_baselines_test/tables/11_baseline_comparison_summary.csv`
- Tables: `/home/mandrakedrink/projects/research_ts_bs/docs/08d_predicted_roi_random_and_domain_matched_baselines_test/tables`
- Reports: `/home/mandrakedrink/projects/research_ts_bs/docs/08d_predicted_roi_random_and_domain_matched_baselines_test/reports`
