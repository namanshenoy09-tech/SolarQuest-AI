# Madrid dataset attribution

The Madrid quality-check CSVs and evaluation reports are derived from:

Oscar Perpiñán-Lamigueiro and Julio Amador-Guerra,
“Electrical power measurements from multiple grid-connected photovoltaic
systems on the rooftop of ETSIDI-UPM (Madrid, Spain),” version 1.0,
2026. https://doi.org/10.5281/zenodo.19652700

The dataset is licensed under Creative Commons Attribution 4.0:
https://creativecommons.org/licenses/by/4.0/

The quality-check CSVs contain quality-filtered hourly sampled mean power
for the T1_S1_Pac channel. Missing and rejected hours are not filled with
zero. The analyses also use Open-Meteo Previous Runs weather data; those
downloaded weather files are not included in this commit. Users should
review the applicable Open-Meteo terms before commercial use.

These evaluations are exploratory. Meter/weather timing alignment remains
provisional; the results are not operational forecast validation.
