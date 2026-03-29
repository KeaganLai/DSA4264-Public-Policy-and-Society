# Improved RDD Summary (School FE + Clustered SE)

Data source: ..\data\hdb_nearest_sch.csv
Outcome: log_real_price_psf
Main bandwidth: 0.5 km

## Main specs (threshold=80)
- schoolfe_adjusted | cutoff=1.0 km | coef=-0.0087 | se=0.0065 | p=0.1831 | pct=-0.87% | n=134487 | clusters=53
- schoolfe_minimal | cutoff=1.0 km | coef=-0.0171 | se=0.0111 | p=0.1251 | pct=-1.69% | n=134487 | clusters=53
- schoolfe_adjusted | cutoff=2.0 km | coef=-0.0074 | se=0.0068 | p=0.2745 | pct=-0.74% | n=69786 | clusters=45
- schoolfe_minimal | cutoff=2.0 km | coef=0.0060 | se=0.0128 | p=0.6409 | pct=0.60% | n=69786 | clusters=45

## Files generated
- rdd_schoolfe_bandwidth_sensitivity.csv
- rdd_schoolfe_covariate_balance.csv
- rdd_schoolfe_donut_results.csv
- rdd_schoolfe_main_results.csv
- rdd_schoolfe_placebo_results.csv
- rdd_schoolfe_summary.md
- rdd_schoolfe_threshold_results.csv