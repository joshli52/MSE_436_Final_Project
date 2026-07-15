# MSE_436_Final_Project
Conference Host City Intelligent Decision Support System (IDSS)
Project Overview

This IDSS helps conference organizers select the optimal U.S. conference host city before venue booking decisions are made.

Instead of relying only on historical airfare or intuition, the system predicts future travel conditions for each candidate city and combines multiple model outputs into a single decision score. The highest-ranked city is recommended to the planner.

The system supports one decision: **Which U.S. city should be selected as the conference host city?**

## Setup

```bash
pip install -r requirements.txt
brew install libomp          # macOS only, required by LightGBM
```

## Running the pipeline

```bash
# Steps 1 & 2 are 
# 1. Ingest data and build features (downloads ~120k rows, caches on disk)
python3 src/build.py

# 2. Train models and generate plots
python3 src/model.py

# 3. Run the optimizer
python3 src/cli.py --sources '[{"city":"Chicago, IL","attendees":30},{"city":"Dallas/Fort Worth, TX","attendees":50}]'

All output files land in the project root:
- `pairs_2025q4.parquet` — 1,000-pair feature table
- `city_coords.parquet` — 115-city geocode + aggregate table
- `distance_calibration.json` — OLS coefficients for gc_km → nsmiles
- `models/` — pickled models, metrics JSON, shipped.json
- `plots/` — pred vs actual, residual vs distance, residual by city

---

## Model architecture

| Model | Description |
|-------|-------------|
| **Baseline 1** | OLS: `fare ~ calibrated_distance` |
| **Baseline 2** | Ridge: `fare ~ log(calibrated_distance) + city dummies` (115 symmetric binary columns) |
| **LightGBM** | Gradient boosting on 15 inference-safe features; depth ≤ 4; city-level LOO aggregates refit inside each CV fold |

**Distance note**: `nsmiles` (actual FAA nonstop distance) is unavailable for
unobserved pairs. `calibrated_distance = 413.98 + 0.3698 × gc_km` is used as
its proxy at inference time (R² = 0.44 for the calibration; see Limitations).

---

## Metrics table

All CV results on 2025-Q4 data (1,000 pairs, 115 cities).
Fares are one-way averages in USD (mean $250, median $242).

### Random 5-fold (interpolation between known markets)

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|-------|--------:|---------:|---------:|---:|
| Baseline 1 (OLS) | 42.55 | 54.51 | 18.6 | 0.152 |
| Baseline 2 (Ridge + city FE) | 37.45 | 47.76 | 16.2 | 0.349 |
| **LightGBM** | **33.42** | **43.49** | **14.1** | **0.460** |


## Optimizer output columns

| Column | Meaning |
|--------|---------|
| `host` | Candidate host city |
| `total_cost` | Sum of attendees × 2 × one-way fare across all sources |
| `cost_per_attendee` | total_cost ÷ total attendees |

---

## Limitations

1. **15.3% pair coverage**: The DOT Table 1 covers only the 1,000 busiest
   markets out of 6,555 possible pairs among 115 cities. All other prices are
   model predictions, not observed data. The optimizer flags each imputed leg.

2. **Contiguous US only**: The dataset excludes Hawaii, Alaska, Puerto Rico,
   and international routes. The 115 cities are all contiguous-state markets.

3. **Quarterly average fares are not bookable prices**: `fare` is the DOT
   quarterly average one-way market fare including all carriers. It does not
   reflect advance purchase, specific dates, seat class, or current inventory.
   Actual booking prices will differ — sometimes substantially.

4. **Round-trip assumption**: The optimizer doubles the one-way fare for
   round-trip cost. DOT round-trip fares can differ from 2× one-way due to
   discount structures. This is a first-order approximation.

5. **Cold-city degradation**: LightGBM R² drops to −1.77 when cities are
   held out of training (cold-city CV). Even Baseline 2 drops from R²=0.35 to
   0.21. Any pair involving a city not in this 115-city dataset should be
   treated as highly uncertain.

6. **Noisy geocodes**: Three cities (Charlottesville VA, New Haven CT, Vero
   Beach FL) have no geocoded rows in the DOT dataset and use hardcoded
   coordinates. Several other cities have incorrect geocodes in the source
   data (e.g., Miami is geocoded to Minnesota), explaining the low
   great-circle → nsmiles calibration R² of 0.44. `nsmiles` (FAA actual
   distance) is used for all observed pairs; `calibrated_distance` is only
   used for unobserved pairs at inference.
