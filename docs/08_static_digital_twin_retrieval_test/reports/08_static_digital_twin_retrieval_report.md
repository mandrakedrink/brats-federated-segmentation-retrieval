# 08 Static Digital Twin Retrieval Report

## Evaluation setup

- Manifest: `/home/mandrakedrink/projects/research_ts_bs/data/processed/manifest_4clients_seed42.csv`
- Split: `['test']`
- Patients: `211`
- ROI source: `gt_wt_center`
- Patch size: `(128, 128, 128)`
- Embedding method: `multiscale_gap_gmp`
- Retrieval spaces: `['embedding', 'tumor_features', 'hybrid']`
- Top-k: `[1, 3, 5, 10]`

## Domain counts

| Domain    |   Patients |
|:----------|-----------:|
| TCGA-GBM  |         20 |
| TCGA-LGG  |         13 |
| UCSF-PDGM |         76 |
| UPENN-GBM |        102 |

## Experiments

| Key                 | Model                                            |
|:--------------------|:-------------------------------------------------|
| centralized_s32_cew | Centralized SegResNet32 final CE-weighted — TEST |
| fedavg_s32_cew      | FedAvg SegResNet32 final CE-weighted — TEST      |
| fedprox_s32_cew     | FedProx SegResNet32 final CE-weighted — TEST     |

## Encoder embedding retrieval at k=5

| Model                                            |   ET consistency |   ET volume diff |   Tumor z-distance |   Same-domain rate |   Same-domain enrichment |
|:-------------------------------------------------|-----------------:|-----------------:|-------------------:|-------------------:|-------------------------:|
| Centralized SegResNet32 final CE-weighted — TEST |           0.9848 |           0.7504 |             1.1972 |             0.6796 |                   2.4123 |
| FedAvg SegResNet32 final CE-weighted — TEST      |           0.9848 |           0.6867 |             1.1566 |             0.5697 |                   1.7595 |
| FedProx SegResNet32 final CE-weighted — TEST     |           0.981  |           0.7533 |             1.1845 |             0.5545 |                   1.7269 |

## Evaluation note

ROI localization uses the ground-truth WT center. This analysis therefore represents the GT-centered upper-bound retrieval setting rather than the primary predicted-mask ROI pipeline.

## Metric notes

- `embedding` evaluates nearest-neighbor retrieval in the segmentation-trained encoder representation.
- `tumor_features` is a phenotype-feature retrieval baseline and is not a learned embedding.
- `hybrid` combines representation and tumor-feature information and is reported separately.
- Same-domain rate and enrichment describe domain clustering; they are not retrieval-quality metrics by themselves.

## Outputs

- Tables: `/home/mandrakedrink/projects/research_ts_bs/docs/08_static_digital_twin_retrieval_test/tables`
- Embeddings: `/home/mandrakedrink/projects/research_ts_bs/docs/08_static_digital_twin_retrieval_test/embeddings`
- Figures: `/home/mandrakedrink/projects/research_ts_bs/docs/08_static_digital_twin_retrieval_test/figures`
- Reports: `/home/mandrakedrink/projects/research_ts_bs/docs/08_static_digital_twin_retrieval_test/reports`
