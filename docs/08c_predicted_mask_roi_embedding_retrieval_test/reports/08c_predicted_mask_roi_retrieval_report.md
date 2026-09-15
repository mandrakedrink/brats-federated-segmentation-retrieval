# 08c Predicted-mask ROI Static Digital Twin Retrieval Report

## Evaluation setup

- Split: `['test']`
- Patients: `211`
- ROI source: `predicted_wt_center`
- Fallback if predicted WT is empty: `image_foreground_center`
- GT fallback allowed: `False`
- Retrieval space used for the primary summary: `embedding`
- k: `5`

## Domain counts

| Domain    |   Patients |
|:----------|-----------:|
| TCGA-GBM  |         20 |
| TCGA-LGG  |         13 |
| UCSF-PDGM |         76 |
| UPENN-GBM |        102 |

## Predicted-ROI embedding retrieval at k=5

| Model                                            |   ET consistency |   WT volume diff |   TC volume diff |   ET volume diff |   Tumor z-distance |   Same-domain rate |   Same-domain enrichment |   No-WT fallback rate |   Mean center distance |   Median center distance |
|:-------------------------------------------------|-----------------:|-----------------:|-----------------:|-----------------:|-------------------:|-------------------:|-------------------------:|----------------------:|-----------------------:|-------------------------:|
| Centralized SegResNet32 final CE-weighted — TEST |           0.9848 |           0.5715 |           0.633  |           0.7549 |             1.2    |             0.6711 |                   2.4306 |                     0 |                  1.937 |                    1.414 |
| FedAvg SegResNet32 final CE-weighted — TEST      |           0.9848 |           0.4693 |           0.587  |           0.6925 |             1.1561 |             0.5678 |                   1.7661 |                     0 |                  2.468 |                    1.414 |
| FedProx SegResNet32 final CE-weighted — TEST     |           0.981  |           0.4996 |           0.5998 |           0.7608 |             1.1799 |             0.5611 |                   1.7434 |                     0 |                  2.53  |                    1.414 |

## Predicted ROI vs GT-centered ROI

Values are predicted-ROI metrics minus the corresponding GT-centered ROI metrics.

| Experiment          |   WT diff delta |   TC diff delta |   ET diff delta |   Tumor z-distance delta |   Same-domain rate delta |
|:--------------------|----------------:|----------------:|----------------:|-------------------------:|-------------------------:|
| centralized_s32_cew |          0.003  |          0.0019 |          0.0045 |                   0.0028 |                  -0.0085 |
| fedavg_s32_cew      |          0.0162 |          0.0096 |          0.0057 |                  -0.0005 |                  -0.0019 |
| fedprox_s32_cew     |         -0.01   |          0.0006 |          0.0075 |                  -0.0045 |                   0.0066 |

## Method notes

- Predicted WT determines ROI localization; ground-truth segmentation is not used to select the ROI.
- Ground-truth tumor features are used only to evaluate phenotype similarity of retrieved neighbors.
- `embedding` uses the SegResNet encoder representation.
- `pred_tumor_features` uses phenotype features derived from the predicted segmentation mask.
- `hybrid_pred_features` combines the encoder representation with prediction-derived tumor features.
- Predicted-to-GT center distance is reported as an ROI localization diagnostic.

## Outputs

- Tables: `/home/mandrakedrink/projects/research_ts_bs/docs/08c_predicted_mask_roi_embedding_retrieval_test/tables`
- Embeddings: `/home/mandrakedrink/projects/research_ts_bs/docs/08c_predicted_mask_roi_embedding_retrieval_test/embeddings`
- Reports: `/home/mandrakedrink/projects/research_ts_bs/docs/08c_predicted_mask_roi_embedding_retrieval_test/reports`
