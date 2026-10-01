# Dataset attribution and generated-data limits

Outdoor recordings and derived histories/traces: **Komiljon Askarov and Jae-ho
Choi (2024)**, *Data on different particulate matter profiles produced in laboratory
from construction activity and outdoor monitoring*, Mendeley V1,
[DOI 10.17632/7f22n9v7hp.1](https://data.mendeley.com/datasets/7f22n9v7hp/1).

Hourly station records and derivatives: **Song Chen (2017)**, *Beijing Multi-Site
Air Quality*, UCI,
[DOI 10.24432/C5RK5G](https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata).

Both are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Authors have not
endorsed this prototype. Changes: causal resampling/lags/rolling features, blocked
splits, model fitting, request histories and forecast traces. Hourly RH is derived
by the declared Magnus approximation; wind compass sectors become wind-to east/
north components. No backward filling or invented alignment between datasets.
Exact source/recording hashes: `reports/datasets.json`. Bulk original data is not tracked.

Plume/efficiency/spraying data is **generated** from the repository's explicit toy
equations/seeds. It contains no measured treatment, droplet-efficiency, perimeter
arrival or fault-diagnosis labels. Code/models follow the repository MIT license;
measurement-data attribution remains as above.
