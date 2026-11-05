# Facial temperature (FLIR AX8): method and quality control

This folder documents how the facial temperature series in `derivatives/facial_temperature/sub-*/ses-*/` were obtained from the thermal camera recordings, and how they were validated. The raw video and snapshots show faces and are not distributed.

**Recommended use:** relative, within-session change of a participant's head temperature, using `T_median` or its background-referenced form `T_median_minus_bg`. The values are not absolute skin temperatures, and `T_max` should not be used to compare participants (see *Key findings*).

## Recordings

A FLIR AX8 on a fixed mount, about 1.5 m from the workstations and facing both, recorded each session (factory settings: emissivity 0.95, reflected temperature 20 °C). The camera stream was not recorded radiometrically: the recording software saved rendered frames, and one lossless PNG snapshot about every 10 s. Each frame shows an iron colour palette with MSX visual-edge overlay, an auto-ranging temperature scale, a crosshair and text. The only absolute information in the image is the scale's MAX and MIN labels and the spot meter at the crosshair.

In total: 12 recording folders, 18 streams, 3,522 snapshots of 640 × 480 pixels.

Recordings are identified by their original folder name, which names the participants and session recorded. For example, `P01A-P02A` is sub-P01/ses-A and sub-P02/ses-A (two seats), and `P10A` is sub-P10/ses-A (one seat). Short forms such as `P01B` below refer to these folders. `thermal_coverage.tsv` (one level up) links every folder and seat to the participant and session.

## Method

1. **Time base.** Snapshot file times survive at 2-s resolution. For each stream a linear fit of file time against snapshot index gives `time_local` (residual SD about 0.6 s, maximum 2.1 s). Very short or irregular streams (the P05A pre-session reconnections) use the file time directly (`time_method`). The snapshot interval was 10.06–10.08 s, except in the first P01A-P02A stream (5.06 s). `time_nominal` (file-name time + 10 s per snapshot) is kept for comparison; it lags the corrected time by up to about 40 s at the end of a session.
2. **Reading the scale.** The MAX, MIN and spot labels were read by template matching against digit glyphs cut from 14 visually verified frames. A label was accepted only if every digit matched with normalised cross-correlation ≥ 0.80, the margin to the next-best digit was ≥ 0.04, the decimal point was present, and there was no ink outside the digit positions. Rejected labels are left missing, never guessed.
3. **Palette inversion.** The colour bar defines the palette: 236 distinct colour levels between MIN (bottom) and MAX (top), so one level is about 0.05 °C at a 10 °C span. Every pixel colour is mapped to the nearest level in CIELAB colour space, giving T = MIN + (MAX − MIN) × position. Pixels are masked when no palette colour is close (ΔE > 12; MSX edges, compression artefacts) or when they lie under a fixed overlay. A frame is rejected if its colour bar differs from the reference palette (median ΔE > 20).
4. **Spot-meter check.** In every frame, the inverted temperature at the crosshair is compared with the camera's own spot reading. A frame fails (`spot_check`) when its difference departs by more than 0.5 °C from the stream's running median; this happens when image and labels are out of sync (16 frames).
5. **Camera position.** The camera stayed still within four positions (`pose` A–D; residual shift ≤ 3 px for 95 % of frames). It was moved once, between snapshots 398 and 399 of P01A-P02A.
6. **Head region.** Each camera position has a fixed search box per seat (L, R; C for the single central seat used by sub-P10). Within the box, the head region is the connected area within 3 °C of the box maximum, after a 5 × 5 median filter that removes MSX edge lines.
7. **Occupancy.** A seat counts as occupied when the warmest part of its box is more than 3.91 °C above the frame median (threshold from Otsu's method over all seat-frames). Single-snapshot changes are smoothed.
8. **Other person present.** Frames in which someone other than the participants (usually the experimenter) is in view are flagged, and the seat values in those frames are marked invalid.
9. **Vacancy events.** Runs of unoccupied snapshots lasting at least 30 s on a used seat.

## Validation

All numbers are in `thermal_validation.json`.

| Check | Result |
|---|---|
| Label reading, random sample (72 frames, 216 labels), compared with visual transcription | 100 % exact |
| Label reading, sample of rare digits (15 frames) | 100 % exact |
| Labels accepted | MAX and MIN read in 3,519 of 3,522 frames |
| Label consistency | MIN < MAX in all frames; MAX 31.8–35.9 °C, MIN 22.6–25.7 °C; no jumps > 1.5 °C |
| Inverted vs camera spot reading, all frames (n = 3,509) | bias +0.09 °C, 95 % limits of agreement −0.41 to +0.60 °C |
| … spot on background (n = 3,432) | bias +0.11 °C, limits −0.23 to +0.46 °C |
| … spot on a warm target (n = 77) | median −0.53 °C |
| Occupancy vs visual review of all 7,030 seat-frames | accuracy 99.94 %, sensitivity 100 %, specificity 99.6 % |
| Other-person flag vs 35 visually noted episodes | 32 detected |

## Key findings

- **`T_max` saturates.** The top 2.9 % of the palette (about 0.3 °C) is a single white colour, and in 98 % of valid frames the face reaches it. `T_max` and `T_p95` are therefore nearly identical for both seats and should not be used to compare participants.
- **`T_median` does not saturate** and is the recommended measure.
- **Camera offset changes.** The camera's reading of the background varies by about 0.25 °C between snapshots and occasionally jumps by ≥ 0.5 °C (88 frames; `offset_step`), for example at a shutter calibration event. `T_median_minus_bg` removes these common changes, but also any genuine change in room-surface temperature; report which form you use.
- **Seats were not vacated during sessions.** A visual review confirmed that participants stayed seated from the first to the last snapshot; the only vacancy is at the end of P08A-P09A.

## Limitations

- **Not radiometric.** Values come from 8-bit, compressed, edge-blended images. On background they agree with the camera's spot reading within about ±0.4 °C; on warm targets, where faces are, they read about 0.5 °C low.
- **Camera accuracy** is ±2 °C or ±2 % (manufacturer specification), and the emissivity was left at 0.95 rather than the ≈0.98 of skin.
- **Resolution.** The sensor has 80 × 60 pixels, so a face spans about 8–12 sensor pixels. Smaller facial regions (e.g. the nose tip) cannot be measured reliably and are not provided.
- **Head region.** It can include the neck, and hands touching the face can become the maximum. In P01A (camera position A) the left participant is cut at the image edge; for sub-P10 the crosshair overlaps the face (crosshair pixels are masked).
- **Other-person flag** misses 3 of 35 episodes, notably P01B snapshots 64–72.
- **Baseline:** a camera failure (10:18–10:36) removed the baseline of sub-P05/ses-A and sub-P06/ses-A (`baseline_usable` in `thermal_coverage.tsv`).

## Files

| File | Content |
|---|---|
| `thermal_frames.tsv` | One row per snapshot (3,522): time, camera position, scale labels, frame checks, background temperature, quality flags |
| `thermal_streams.tsv` | One row per recording stream (18): timing fit and video-file evidence |
| `seat_vacancy_events.tsv` | Vacancy intervals of at least 30 s |
| `other_person_episodes.tsv` | Episodes with another person in view, with the nearest seat |
| `ocr_manual_check.tsv` | Visual transcription vs automatic reading, per label, for the validation samples |
| `occupancy_visual_census.tsv` | Visual occupancy label vs automatic classification for every seat-frame |
| `thermal_validation.json` | All validation statistics |
| `figures/` | Spot-meter agreement, per-recording seat time series, palette, timing drift, occupancy-score distribution (no images of faces) |

Every table has a JSON file of the same name describing its columns.
