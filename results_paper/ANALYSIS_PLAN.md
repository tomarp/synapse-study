# SYNAPSE results paper — analysis plan

Registered by committing this file to https://github.com/tomarp/synapse-study **before any comparison between
experimental conditions was made**. Until this commit, the data had been examined only for structure, coverage,
data quality and pooled distributions (curation and the Data Descriptor's technical validation); no outcome had been
summarised by lighting or soundscape condition by anyone.

| | |
|---|---|
| Study | SYNAPSE (DTU, 24–26 June 2025), 10 participants × 2 sessions |
| Data | published dataset v1.0.0, https://doi.org/10.5281/zenodo.23458777 (no raw data) |
| Target journal | *Building and Environment* |
| Authors | P. Tomar, T. Dave, P. Wargocki, A. L. Pisello |
| Original aim of the study (2025) | effects of lighting colour temperature and soundscapes on cognitive performance |
| Framework | Bayesian estimation with hierarchical (mixed) models; no significance testing |

## 1. Research questions

- **RQ1 — cognitive performance (co-primary).** How do correlated colour temperature (cool 6250 K vs warm 2014 K
  luminaire setpoint) and urban soundscape (road traffic vs cafeteria babble, over headphones) change response speed
  and accuracy in working-memory, interference-control and selective-attention tasks?
- **RQ2 — perceived indoor environment (co-primary).** How do the two factors change the perception of the visual and
  acoustic environment (manipulation checks), and do they change the perception of the other domains — thermal and air
  quality — that were held nearly constant (cross-modal effects)?
- **RQ3 — affect and workload (secondary).** How do the factors change task-related affect, alertness and workload?
- **RQ4 — physiology (secondary).** How do the factors change heart rate, resting heart-rate variability, wrist skin
  temperature and facial temperature?
- **RQ5 — individual differences (secondary).** How much do participants differ in their response to each factor?

Expected directions, stated in advance (they do not change the analysis): cool light faster responses and brighter
appearance; warm light warmer thermal sensation (hue–heat hypothesis); babble (intelligible speech) poorer
working-memory performance than traffic (irrelevant-speech effect); traffic (+4.2 dB(A)) noisier and less acceptable
acoustics than babble. All estimands are reported whatever their direction.

## 2. Design and identification

Lighting varied between a participant's two sessions (same day, 10–15 min apart; cool first for 6, warm first for 4).
The soundscape varied between the two task rounds of a session; its order was reversed in the second session for 9
participants (not for sub-P02). Each task variant was always presented in the same round.

Coding (centred, ±0.5): light `L` (cool +½, warm −½), soundscape `S` (traffic +½, babble −½), session order `O`
(second +½), round `R` (round 2 +½).

| Effect | Identified from | Consequence |
|---|---|---|
| `S` on ratings and physiology | the two rounds of the same session, order counterbalanced | within-session, free of order and lighting |
| `S` on performance | within participant, from the difference between round-1 and round-2 task sets across the two sessions (the soundscape of each task set is reversed while the lighting changes for both) | within-participant; assumes the lighting effect is the same for both task sets; sub-P02 contributes no information |
| `L` | the two sessions of a participant | within-participant, separated from session order `O` only between participants (6 vs 4) |
| `L × S` | between the two soundscape-pairing groups (5 vs 4 participants) | weakly identified; reported as secondary, never interpreted on its own |
| `O`, `R` | nuisance; always in the model | — |

The screen colour theme followed the lighting (cool: dark, warm: light) and cannot be separated from `L`; lighting
effects are therefore effects of the lighting condition as delivered (luminaire preset + screen theme). Daylight entered
the room; room illuminance (uncalibrated monitor, per-session median) is used in a sensitivity analysis.

## 3. Outcomes and data preparation

General: rows with `superseded_run = true` are never used; missing values are not imputed; all labels are taken from
the dataset (`light` from `sub-*_sessions.tsv`, `soundscape` and `round` from the task tables; `people` = babble).

### Family A — cognitive performance (co-primary, RQ1)

Tasks: `nbackletter`, `nbackdigit`, `flankerarrow`, `flankerdigit`, `stroopword`, `stroopemoji` (three per round).
Not used: `visualcount` (no valid answer key), `visualshape` (keys partly indeterminate; descriptive only),
`stroopemotionalword` and `stroopemotionalimage` (early configuration, one session each).

- **A1 Speed:** trial-level log response time of correct trials with `rt_flag = ok`.
- **A2 Accuracy:** trial-level correctness (`correct`) of trials with `rt_flag = ok`.
- **Pre-specified block exclusions (primary analysis):** task blocks with a documented response strategy that differs
  from the instruction (dataset README): `stroopemoji` in sub-P04/ses-A, sub-P06/ses-A, sub-P06/ses-B, sub-P09/ses-A,
  sub-P09/ses-B (word named instead of ink colour) and `flankerdigit` in sub-P04/ses-A (reversed parity mapping).
- **Trial type:** each task × trial-type cell has its own intercept (n-back level × target/non-target; flanker
  congruent/incongruent/neutral; Stroop one cell). Within-block trial position (log of trial index, centred per task) is
  a covariate.

### Family B — perceived indoor environment (co-primary, RQ2)

Eight 5-point items at T1 (after the 5-min baseline, no soundscape), T2 and T3 (after rounds 1 and 2):
`thermal_sensation`, `thermal_satisfaction`, `air_quality_assessment`, `air_quality_satisfaction`, `light_assessment`,
`light_satisfaction`, `acoustic_assessment`, `acoustic_satisfaction` (60 ratings each). Manipulation checks: `L` on the
two light items, `S` on the two acoustic items. Cross-modal estimands: `L` on the thermal, air and acoustic items;
`S` on the thermal, air and light items. Preference items (3-level) are secondary.

### Family C — affect and workload (secondary, RQ3)

T2 and T3 items (9-point; focus 7-point): `task_difficulty`, `task_focus`, `relaxation_alertness`,
`emotional_stability`, `task_engagement`, `cognitive_load`, `emotional_valence`, `emotional_arousal` (40 ratings each).

### Family D — physiology (secondary, RQ4)

Phases from `derivatives/session_timeline/timeline.tsv`: baseline (5 min), round 1, round 2 (soundscape switch
excluded). sub-P06 has no wrist data.

- **D1 Heart rate:** wrist inter-beat intervals (`desc-ibi`), kept if 0.33–1.5 s and within 20 % of the previous
  accepted interval; phase mean heart rate (bpm); a phase needs ≥ 50 % of its expected beats.
- **D2 Resting HRV:** ln RMSSD of the accepted intervals of the baseline (participant still, no typing); same coverage rule.
- **D3 Wrist skin temperature:** phase mean (°C).
- **D4 Facial temperature:** phase mean of `T_median_minus_bg` over `valid = true` snapshots (relative values: only
  within-session contrasts are estimated; `T_median` as sensitivity).

## 4. Models

All models: PyMC, NUTS (nutpie), 4 chains × 1000 draws after 1000 tuning steps, fixed seed. Participant random
intercepts throughout. Priors are weakly informative on the scale of each outcome.

| Model | Likelihood | Linear predictor (besides cell / threshold intercepts) | Effect priors |
|---|---|---|---|
| A1 | Normal on log RT, task-specific residual SD | `L + S + O + L×S + trial position`; task-specific deviations of `L` and `S` (partial pooling, SD ~ HalfNormal(0.05)); participant random slopes for `L` and `S`; participant × task intercepts | Normal(0, 0.1) |
| A2 | Bernoulli-logit | as A1 (no residual SD) | Normal(0, 0.5) |
| B, C (one model per item) | cumulative logit (ordered thresholds) | B: `L + S·snd + snd + R·snd + O + L×S·snd`, where `snd` = 1 at T2/T3, 0 at T1; C: `L + S + R + O + L×S` | Normal(0, 1) |
| D1–D3 | Normal | `task + L + S·task + R·task + O + L×S·task` (`task` = 1 in rounds); participant and session random intercepts | D1 Normal(0, 5) bpm; D2 Normal(0, 0.3); D3 Normal(0, 0.5) °C |
| D4 | Normal | session intercepts (absorb between-session offsets) + `task + S·task + R·task` | Normal(0, 0.5) °C |

Primary estimands: `β_L` and `β_S` in A1, A2 and B (task-averaged in A). Reported as: A1 percent change in response time
(100·(e^β − 1)); A2 odds ratio of a correct response; B and C the latent log-odds and the difference in expected rating
(scale points) for an average participant; D in outcome units. Secondary estimands: task-specific effects (A),
`L × S`, preference items, families C and D, and the between-participant SD of the `L` and `S` effects (RQ5).

## 5. Inference and reporting

For each estimand: posterior median, 95 % equal-tailed credible interval (CrI), probability of direction (pd) and the
posterior probability that the effect lies outside a region of practical equivalence (ROPE): log RT ±0.02 (≈ 2 %),
accuracy ±0.2 log-odds, ratings ±0.18 log-odds (standardised 0.1), heart rate ±1 bpm, ln RMSSD ±0.05, skin and facial
temperature ±0.1 °C.

Wording: *credible* when the 95 % CrI excludes zero; *probable* when pd ≥ 0.90; *practically negligible* when ≥ 95 % of
the posterior lies inside the ROPE; otherwise *inconclusive*. All estimands of a family are reported together
(tables, forest plots), whatever their result; there is no selection of outcomes.

Convergence: R̂ ≤ 1.01, bulk ESS ≥ 400, no divergent transitions; otherwise the model is reparametrised or tuned longer
and the change reported. Posterior predictive checks are shown in the supplement.

## 6. Sensitivity analyses

| # | Analysis | Families |
|---|---|---|
| S1 | exclude the early-configuration sessions sub-P01/ses-A and sub-P02/ses-A | A |
| S2 | include the response-strategy blocks | A |
| S3 | A1 with all `rt_flag = ok` trials (correct and incorrect) | A |
| S4 | exclude questionnaires with a weak participant link (`link_quality`) | B, C |
| S5 | add the session-median room illuminance (standardised) as a covariate (daylight) | A, B, D |
| S6 | add a random intercept for the testing slot (participants tested together) | A, B |
| S7 | effect priors twice and half as wide | primary estimands |
| S8 | one joint model over the eight B items with partially pooled cross-modal effects | B |
| S9 | facial temperature from `T_median` | D4 |

## 7. Exploratory analyses (reported as exploratory)

Room conditions (air temperature, humidity, CO₂, sound level, illuminance) by lighting condition and round, as a check
that the non-manipulated domains did not differ; within-person associations between round-level ratings, performance
and physiology (e.g. acoustic satisfaction and response time; heart rate and arousal); participant-level effect
estimates; visual-task performance descriptively.

## 8. Blind analysis

The pipeline is written and checked (data extraction, model convergence, prior and posterior predictive checks, figures)
on data with **scrambled condition labels**: for every participant the lighting labels of the two sessions, and for every
session the soundscape labels of the two rounds, are swapped with probability ½ (fixed seed). The pipeline is then
frozen and committed, and run once on the true labels. Any change after unblinding is reported as a deviation.

## 9. Software

Python 3.12; PyMC 6.4.0, nutpie 0.16.11, ArviZ 1.3.0, pandas, NumPy, SciPy, Matplotlib (exact versions in
`requirements.txt`). Code: `results_paper/` of this repository.
