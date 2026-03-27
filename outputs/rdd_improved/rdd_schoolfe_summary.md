# Improved RDD Summary (School FE + Clustered SE)

Data source: ..\data\hdb_nearest_sch.csv
Outcome: log_real_price_psf
Main bandwidth: 0.5 km

## Main specs (threshold=75)
- schoolfe_adjusted | cutoff=1.0 km | coef=-0.0054 | se=0.0055 | p=0.328 | pct=-0.53% | n=156398 | clusters=63
- schoolfe_minimal | cutoff=1.0 km | coef=-0.0077 | se=0.0124 | p=0.5343 | pct=-0.77% | n=156398 | clusters=63
- schoolfe_adjusted | cutoff=2.0 km | coef=0.0017 | se=0.0090 | p=0.8462 | pct=0.17% | n=56378 | clusters=51
- schoolfe_minimal | cutoff=2.0 km | coef=0.0551 | se=0.0417 | p=0.187 | pct=5.66% | n=56378 | clusters=51

## Files generated
- rdd_schoolfe_bandwidth_sensitivity.csv
- rdd_schoolfe_covariate_balance.csv
- rdd_schoolfe_donut_results.csv
- rdd_schoolfe_main_results.csv
- rdd_schoolfe_threshold_results.csv