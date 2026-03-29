# RDD Summary

Data source: ..\data\hdb_nearest_sch.csv
Outcome: log_real_price_psf
Main bandwidth: 0.5 km

## Main specs (threshold=80)
- main_adjusted | cutoff=1.0 km | coef=-0.0032 | se=0.0013 | p=0.01464 | pct=-0.32% | n=134487
- main_minimal | cutoff=1.0 km | coef=-0.0183 | se=0.0024 | p=1.435e-14 | pct=-1.82% | n=134487
- main_adjusted | cutoff=2.0 km | coef=0.0002 | se=0.0019 | p=0.9362 | pct=0.02% | n=69786
- main_minimal | cutoff=2.0 km | coef=-0.0436 | se=0.0041 | p=7.124e-26 | pct=-4.27% | n=69786

## Files generated
- rdd_bandwidth_sensitivity.csv
- rdd_covariate_balance.csv
- rdd_density_proxy.csv
- rdd_donut_results.csv
- rdd_main_results.csv
- rdd_placebo_results.csv
- rdd_plot_threshold75_cutoff1km.png
- rdd_plot_threshold75_cutoff2km.png
- rdd_plot_threshold80_cutoff1km.png
- rdd_plot_threshold80_cutoff2km.png
- rdd_summary.md
- rdd_threshold_results.csv