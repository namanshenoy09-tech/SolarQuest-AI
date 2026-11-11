# Cyprus: March-plus-May training comparison on a July holdout

## Registration and prior inspection
- In this workflow, Cyprus May quality and forecast scores and March/April
  quality counts were inspected before this protocol. No July model scores
  were calculated. July data and weather coverage remain unverified.
- Dataset: https://doi.org/10.5281/zenodo.20800202.
- March file 2025_03.csv: publisher MD5 78e899e964dd46b549459e9c7481bb04;
  separately downloaded copy SHA-256
  91b55174e08bb4ec5b269cd2881357b78018b7f22c9ae8472d690dfce53dda33.
- July file 2025_07.csv: publisher MD5 22f5db59ca74c8b5d2a8aef5f56065e2.
- Original May protocol SHA-256: 2a0dfc8f866d627690316f8a37860d045e1c4cd54b8af48c3f47ad497fa7d8c6
- Original May holdout report SHA-256: 4f0081ad2892ff40a58fbdf9aae64a73b82134edb9e5d4407f3c73381251d50c
- Original May feasibility report SHA-256: 1c11b5552800f99fbf965395e47b939656e9c5e00163399916f3a874cea45e44
- Original May evaluator SHA-256: 4a6fcc7afcd64eb27aae2a22ab071d60a7a23ac6be8e1fafcdb458f2b00b892c

## Models and split
- Both models are site-specific ordinary linear regressions with the original
  temperature, shortwave radiation and local-hour sine/cosine features.
  Both clip predictions below zero and force zero at forecast radiation <= 0.
- Model A refits using exactly the original 105 May training rows and archived
  weather files, with local starts before 2025-05-21 23:00. Verify that the
  refit reproduces the saved May paired scores and use the same sklearn version.
- Model B uses those same May rows plus complete matched March 2025 hours
  with local starts >= 2025-03-01 00:00 and < 2025-04-01 00:00.
  There is no April, May holdout, June or July training data.
- Fixed July test: local hour starts >= 2025-07-23 00:00 and < 2025-07-31
  00:00, 192 calendar hours. Do not change dates after seeing results.

## Quality and weather
- Target is the sampled mean of 120 unique, aligned, finite 30-second values
  in W. Missing readings are never filled with zero. Handle the March clock
  change in UTC; retain distinct instants and check source offsets against the
  provisional Asia/Nicosia timezone before using local-hour features.
- Use archived Open-Meteo previous-day2 temperature and shortwave radiation
  for requested coordinates 34.927423, 33.475382, joined to the local hour
  end in UTC, as in May. Save request and returned grid metadata.
- Hour alignment and sampling semantics remain provisional. Previous-day2
  fields are not necessarily a single simultaneously issued 48-hour forecast.

## Comparison and safeguards
- Compare A, B, 72-hour persistence and seven-day persistence using identical
  complete July target hours with weather and both complete baseline readings.
  Baselines assume timely telemetry and use only earlier eligible readings.
- Primary: paired MAE in W for all matched hours. Also report paired MAE for
  positive forecast radiation, descriptive daily scores and coverage.
  Do not interpret MAE reduction as financial savings.
- Require all original 105 matched May rows, at least 50 additional matched
  March training hours and at least 20 paired July hours. These are feasibility
  floors only; they do not establish statistical or operational reliability.
- If inputs or floors fail, report the issue; do not select a different test
  window or alter model rules after seeing July scores. Subsequent tuning
  requires a fresh untouched evaluation period.
- This is exploratory site-specific retraining, not cross-site transfer or
  operational forecast validation. Further sites and true forecast
  issue-time evaluation remain necessary.
