# Cycling Traffic Analysis in Flanders

A modern data analytics project analysing cycling traffic patterns and deviations using bicycle count data from the Flemish AWV counting network.

This project combines data preparation, exploratory analysis, expected-count modelling, deviation detection, and dashboard-based visualisation in one reproducible workflow.

## Overview

This project studies cyclist traffic in Flanders using AWV bicycle count data and external contextual information. Its main objective is to understand normal cycling patterns across time and location, estimate expected cyclist counts, and identify intervals in which observed traffic differs substantially from that expected behaviour.

The analytical workflow follows five main stages:
- loading raw AWV data, metadata, and external data sources,
- preprocessing and aggregating 15-minute cyclist counts into 2-hour intervals,
- exploring the data through descriptive and graphical analysis,
- modelling expected cyclist traffic,
- and detecting deviations during the prediction period.

This project:
- explores temporal and spatial cycling traffic patterns,
- models expected bicycle counts,
- detects deviations from expected behaviour,
- and visualizes findings through an interactive dashboard.

## Technologies

The project is implemented primarily in Python and uses notebook-based analysis throughout the workflow.

Main tools and libraries include:
- Python
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- plotly
- Shiny for Python

These technologies support data cleaning, feature construction, exploratory data analysis, count modelling, deviation analysis, and dashboard development.

## Project Structure

```text
app/        -> dashboard application
data/       -> raw, external, and processed datasets
outputs/    -> figures, tables, and diagnostic results
notebooks/  -> end-to-end analytical workflow
```

### Notebook workflow

- `01_load_data.ipynb`  
  Downloads and reads AWV bicycle count files, site metadata, direction metadata, and external data sources such as fuel prices, weather data, holidays, and major events.

- `02_preprocess_counts.ipynb`  
  Filters cyclist observations, aggregates 15-minute counts to 2-hour intervals, handles incomplete intervals, and builds both the modelling dataset and the prediction dataset.

- `03_exploratory_data_analysis.ipynb`  
  Examines missingness, variable distributions, and the structure of the cyclist count response through descriptive statistics and visualisations.

- `04_model_expected_counts.ipynb`  
  Estimates expected cyclist traffic using count models and model comparison on a chronological training, validation, and prediction split.

- `05_detect_deviations.ipynb`  
  Compares observed and expected cyclist counts and identifies unusually high or low traffic intervals using deviation thresholds.

## Running the Dashboard
Link: https://nerissatruong2510.shinyapps.io/cycling-dashboard/
Install the project dependencies and launch the dashboard from the project root:

```bash
pip install -r requirements.txt
shiny run --reload app/app.py
```

To reproduce the full analytical workflow, run the notebooks in order from data loading to deviation detection.

## Dashboard Features

The dashboard is designed to communicate both the descriptive and model-based parts of the analysis in an interactive format.

Mainly include:
- Traffic overview
- Temporal rhythm
- External information patterns
- Deviation overview
- Spatial patterns

## Authors: Group 33

- Anna Solodka (r0926269)
- Maria Amelie Gonda (r0910823)
- Nerissa Truong (r0823591)
- Yuxi Liu (r1070939)
