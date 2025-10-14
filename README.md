# Road Accident Risk Prediction — End-to-End ML Project

![pexels-pixabay-210182](https://github.com/user-attachments/assets/aaef8460-bee9-453c-8ecc-1c49ba5b4032)

## Overview

This project explores one question:

> *Can we predict the likelihood of road accidents from physical and environmental road conditions alone?*

Using structured data on road geometry, weather, and visibility, I built and tuned multiple machine learning models to estimate accident risk. The final model — a tuned XGBoost regressor — achieved an RMSE of 0.056 and an R² of 0.885, showing strong predictive capability.

## Problem Context

Road safety isn’t just about driver behavior — infrastructure and conditions matter too.
Factors like lighting, curvature, weather, and lane count can dramatically influence accident likelihood.

This project focused on quantifying that risk:

> *Given a set of environmental and road characteristics, estimate the probability (or risk score) of an accident occurring.*

## Workflow Summary

### 1. Data Exploration

The dataset contained records of road conditions and corresponding accident risk levels.

Early analysis showed clear patterns — for instance, night-time and dimly lit conditions correlated strongly with higher risk scores, while wider roads (more lanes) generally indicated lower risk.

### 2. Feature Engineering

The features were cleanly grouped into numeric, categorical, and boolean features. Features used for this exercise were:

`num_lanes`, `curvature`, `speed_limit`, `num_reported_accidents`, `road_type`, `lighting`, `weather`, `time_of_day`, `road_signs_present`, `public_road`, `holiday`, `school_season`

Preprocessing included:
* Scaling numeric features
* One-hot encoding categorical columns
* Preserving boolean indicators as binary flags

### 3. Baseline Modeling
I compared several regression algorithms:

* Linear Regression
* Random Forest
* LightGBM
* XGBoost
* CatBoost
* SVR
Each model was evaluated with **cross-validation** (`RMSE`) to ensure generalization.

### 4. Model Tuning

After identifying `XGBoost` as the most promising, I used **RandomizedSearchCV** for hyperparameter tuning to get an optimized XGBoost model with the following evaluation metrics:
train RMSE: ~0.056        test RMSE: 0.0562
train R²:                 test R²: 0.8855

### Feature Importance Insights
From the tuned XGBoost model, the top predictive drivers were:

1. **Lighting (night, dim, daylight)** – poor lighting was strongly tied to higher risk
2. **Speed limit** – higher limits corresponded to elevated accident probability
3. **Curvature** – sharper turns slightly increased risk
4. **Weather (clear, foggy, rainy)** – rain and fog contributed nonlinearly to risk
5. **Number of reported accidents** – historical frequency proved a strong contextual cue

### Tech Stack
* **Python**: pandas, numpy, scikit-learn
* **Modeling**: XGBoost, LightGBM, CatBoost
* **Visualization**: matplotlib, seaborn, SHAP
* **Validation**: Cross-validation, RMSE, R²

### Insights
* Environmental and structural features alone can predict accident likelihood surprisingly well.
* Lighting conditions were the dominant risk factor.
* Model interpretability (via SHAP and feature importance) was crucial for explaining how each factor influences outcomes.
* Proper preprocessing and consistent validation pipelines made experimentation smooth and reproducible.
