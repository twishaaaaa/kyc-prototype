## Table 1. Experiment 1 results (mean ± SD over 5 repeats)

| Feature set | Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| rule: any consistency check fails | no training | 0.732 | 0.894 | 0.805 | n/a |
| consistency only | LogReg | 0.985 ± 0.005 | 0.817 | 0.893 ± 0.002 | 0.902 ± 0.003 |
| consistency only | RandomForest | 0.989 ± 0.003 | 0.808 ± 0.010 | 0.889 ± 0.006 | 0.915 ± 0.006 |
| consistency only | XGBoost | 0.990 ± 0.003 | 0.797 ± 0.006 | 0.883 ± 0.004 | 0.905 ± 0.005 |
| ocr_quality only | LogReg | 0.530 ± 0.011 | 0.544 ± 0.013 | 0.537 ± 0.011 | 0.518 ± 0.013 |
| ocr_quality only | RandomForest | 0.504 ± 0.011 | 0.516 ± 0.024 | 0.510 ± 0.017 | 0.499 ± 0.011 |
| ocr_quality only | XGBoost | 0.515 ± 0.016 | 0.539 ± 0.020 | 0.527 ± 0.018 | 0.524 ± 0.011 |
| forensic only | LogReg | 0.991 ± 0.003 | 0.884 ± 0.011 | 0.935 ± 0.006 | 0.968 ± 0.004 |
| forensic only | RandomForest | 0.943 ± 0.004 | 0.934 ± 0.004 | 0.939 ± 0.002 | 0.990 ± 0.001 |
| forensic only | XGBoost | 0.980 ± 0.009 | 0.936 ± 0.007 | 0.957 ± 0.007 | 0.994 ± 0.001 |
| consistency + ocr_quality | LogReg | 0.993 ± 0.006 | 0.814 ± 0.003 | 0.895 ± 0.003 | 0.919 ± 0.003 |
| consistency + ocr_quality | RandomForest | 0.956 ± 0.010 | 0.809 ± 0.009 | 0.876 ± 0.005 | 0.918 ± 0.003 |
| consistency + ocr_quality | XGBoost | 0.924 ± 0.013 | 0.808 ± 0.015 | 0.862 ± 0.007 | 0.907 ± 0.006 |
| all document features | LogReg | 1.000 | 0.962 ± 0.007 | 0.981 ± 0.004 | 0.999 ± 0.000 |
| all document features | RandomForest | 0.976 ± 0.009 | 0.953 ± 0.003 | 0.965 ± 0.004 | 0.997 ± 0.001 |
| all document features | XGBoost | 0.986 ± 0.008 | 0.958 ± 0.006 | 0.972 ± 0.006 | 0.998 ± 0.001 |

## Table 2. Detection rate by forgery type (Random Forest) and false-positive rate on genuine cards

| Feature set | A | B | C | Genuine (FPR) |
|---|---|---|---|---|
| all document features | 0.90 | 1.00 | 1.00 | 0.02 |
| consistency + ocr_quality | 0.99 | 0.58 | 0.76 | 0.04 |
| consistency only | 0.99 | 0.58 | 0.77 | 0.01 |
| forensic only | 0.86 | 1.00 | 1.00 | 0.06 |
| ocr_quality only | 0.43 | 0.56 | 0.66 | 0.51 |
| rule: any consistency check fails | 1.00 | 0.75 | 0.90 | 0.33 |

## Table 3. Top 10 Random Forest features

| Feature | Importance |
|---|---|
| for_loglap_dev_expiry | 0.105 |
| for_loglap_std | 0.104 |
| validity_span_ok | 0.083 |
| for_loglap_dev_dob | 0.073 |
| for_loglap_dev_issue | 0.063 |
| for_loglap_maxdev | 0.055 |
| for_loglap_dev_docno | 0.038 |
| for_ink_dev_dob | 0.030 |
| for_charh_std | 0.029 |
| name_sim_given | 0.026 |
