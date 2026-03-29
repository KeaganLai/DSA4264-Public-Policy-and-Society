# RDD Summary

Data source: ..\data\hdb_nearest_sch.csv
Outcome: log_real_price_psf
Main bandwidth: 0.5 km

## Main specs (threshold=75)
- main_adjusted | cutoff=1.0 km | coef=0.0023 | se=0.0012 | p=0.0648 | pct=0.23% | n=156398
- main_minimal | cutoff=1.0 km | coef=-0.0012 | se=0.0023 | p=0.6138 | pct=-0.12% | n=156398
- main_adjusted | cutoff=2.0 km | coef=-0.0027 | se=0.0023 | p=0.2346 | pct=-0.27% | n=56378
- main_minimal | cutoff=2.0 km | coef=0.0643 | se=0.0051 | p=1.173e-36 | pct=6.64% | n=56378

## Files generated
- rdd_bandwidth_sensitivity.csv
- rdd_covariate_balance.csv
- rdd_density_proxy.csv
- rdd_donut_results.csv
- rdd_main_results.csv
- rdd_placebo_results.csv
- rdd_plot_threshold75_cutoff1km.png
- rdd_plot_threshold75_cutoff2km.png
- rdd_summary.md
- rdd_threshold_results.csv