# Road Accident Risk Prediction — A Machine Learning Journey

![pexels-pixabay-210182](https://github.com/user-attachments/assets/aaef8460-bee9-453c-8ecc-1c49ba5b4032)

## Project Overview

Every day, countless drivers navigate roads of varying conditions — some wide and straight, others narrow and winding. Understanding which road conditions are more likely to lead to severe accidents can make a real difference in safety planning and road management.

This project tackles exactly that: predicting road risk severity based on infrastructure, environmental, and situational factors.
It’s a data science and machine learning case study focusing on model experimentation, interpretability, and generalization.

## Objective

To build a machine learning model that predicts road risk severity using features such as road type, lighting, weather, and other contextual conditions.

We wanted not just an accurate model — but one that’s explainable, stable, and generalizes well beyond the training data.

## Data Overview

The dataset captures a mix of numeric, categorical, and boolean features representing the environment and road context:
`num_lanes`, `curvature`, `speed_limit`, `num_reported_accidents`, `road_type`, `lighting`, `weather`, `time_of_day`, `road_signs_present`, `public_road`, `holiday`, `school_season`.

Each record corresponds to a specific road segment and associated contextual conditions, labeled with its risk severity score.

Preprocessing included:
* Scaling numeric features
* One-hot encoding categorical columns
* Preserving boolean indicators as binary flags

Early analysis during data exploration showed clear patterns — for instance, night-time and dimly lit conditions correlated strongly with higher risk scores, while wider roads (more lanes) generally indicated lower risk.


### Modeling Process

Our process followed a classic yet refined machine learning workflow:

**1. Baseline Modeling**
    * Established benchmark performance using several algorithms:
        * Linear Regression
        * Random Forest
        * Gradient Boosting
        * CatBoost
        * XGBoost

**2. Initial Findings**
    * CatBoost emerged as the top performer during baseline evaluation (lowest RMSE on validation).
    * We then used feature importance and SHAP values to interpret CatBoost’s decisions globally and locally.

**3. Generalization Testing**

    * To verify robustness, we conducted cross-validation (CV) RMSE analysis.
    * Interestingly, XGBoost slightly outperformed CatBoost here — with lower mean RMSE and smaller standard deviation across folds.
    * This indicated better stability and generalization, making XGBoost our final model of choice.

### Key Insights
* Lighting conditions and weather consistently ranked among the most influential predictors of risk severity.
* Speed limits had a nonlinear relationship with severity — moderate speeds tended to correspond with fewer severe cases.
* Presence of road signs and school season introduced subtle but meaningful contextual effects, showing how small situational cues can shift risk dynamics.

### Final Model

| Metric                             | Value                             |
| ---------------------------------- | --------------------------------- |
| Model                              | XGBoost Regressor                 |
| Validation RMSE                    | *[insert value]*                  |
| Cross-Validation RMSE (mean ± std) | *[insert mean ± std]*             |
| Test Predictions                   | 172,585 rows (matching test size) |

### Reflections
This project reinforced the importance of balancing accuracy and interpretability.

A model that performs slightly better on validation might still fail to generalize well — and that’s where cross-validation and explainability step in as sanity checks.

Also, SHAP proved invaluable for trust-building — showing exactly how and why certain predictions were made, which is critical for real-world adoption in safety-critical domains.

### What's Next?
* Experiment with ensemble stacking of CatBoost and XGBoost.
* Introduce geospatial and temporal data (e.g., location clusters or time-based effects).
* Explore model deployment via a lightweight web dashboard for real-time risk scoring.
