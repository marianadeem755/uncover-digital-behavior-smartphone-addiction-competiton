# **Uncover Digital Behavior: Predict Smartphone Addiction Competition**

* This project focuses on the **Predicting Smartphone Addiction (PS6E8)** competition and covers the complete machine learning workflow, from data exploration, preprocessing to feature engineering, model development, evaluation, and prediction.

* The repository includes **two notebooks which represents 2 versions of the solution**, where the second version builds with further experimentation and improvements aimed at achieving a higher **Kaggle leaderboard score**.


1.  **`uncover-digital-behavior-smartphone-addiction.ipynb`**: an extensive exploratory data analysis, behavioral pattern discovery, feature engineering, and ensemble modeling notebook.
2.  **`predict-smartphone-addiction-ps6e8-v2.ipynb`**: a focused, deep-learning pipeline built around **RealMLP**, multi-seed stratified cross-validation, leakage-safe target encoding, and probability prediction.

This project uses features such as demographic, lifestyle, academic/work-impact, and smartphone-use signals to estimate the probability that a person is
classified as smartphone addicted or not.

## Project Overview

* Smartphone behavioral signals: daily screentime, social-media usage, gaming, notifications, application opens, sleep, work/study activity, and weekend usage. 
* The purpose of this project is to explore how these signals relate to the competition target and then build strong predictive models to take isnghts about the Data.

The project have 2 notebooks:

### **Notebook 1: Behavioral Analysis & Ensemble Modeling**

The first notebook is designed as a broad end-to-end analytical
workflow. It:

*   Introduces and explore the competition dataset.
*   Examine data types, shapes, summary statistics, and missing values.
*   Handles missing values in numerical and categorical variables.
*   Performs extensive behavioral analysis across age groups and categorical segments.
*   Studies different features and patterns such as screen time, social media, gaming, sleep, notifications, app openings, weekend usage, stress, and academic/work impact.
*   Identifies high-activity and extreme-user profiles.
*   Engineers behavioral features.
*   Undergo fetaure engineering by performing categorical transformations, quantile/binning features, interactions, and numerical behavioralratios.
*   Validates the final machine-learning input for missing and infinite values.
*   Train the models such as XGBoost, LightGBM, and TabM models.
*   Uses stratified 3-fold cross-validation for the ensembling.
*   Blends the tree-based models and TabM.
*   Adds a RealMLP prediction layer to the ensemble.
*   Generates a final competition submission based on best results through ensembling.

### **Notebook 2: RealMLP-Focused Prediction Pipeline**

The second notebook provides a more focused predictive workflow. It:

*   Loads the competition training and testing data.
*   Separates the identifier, target, categorical variables, and numerical variables.
*   Creates leakage-safe engineered features.
*   Uses missing-value indicators and row-level missing counts.
*   Creates behavioral ratios and domain-based features.
*   Creates categorical/frequency representations of selected variables.
*   Applies target encoding inside each cross-validation fold.
*   Builds a custom RealMLP neural tabular classifier.
*   Trains using **5-fold stratified cross-validation**.
*   Produces out-of-fold predictions and test-set probabilities.
*   Saves the final submission file based on best performing results.

## **Dataset**

The notebooks use the competition files:

  File                      Purpose
  ------------------------- ------------------------------------------------------
  `train.csv`               Training data containing the target `addicted_label`
  `test.csv`                Test data without the target
  `sample_submission.csv`   Required submission structure

### Dataset Size

According to the notebooks:

-   Training set: **691,369 rows & 14 columns**
-   Testing set: **296,302 rows & 13 columns**
-   The training data contains the target column `addicted_label`.
-   The test data does not contain the target.
-   The sample submission contains the test IDs and prediction column.

## Features

The core behavioral variables used by the project include:

### Numerical Features

  Feature                     Description
  --------------------------- ---------------------------------------
  `age`                       Age of the individual
  `daily_screen_time_hours`   Average daily smartphone screen time
  `social_media_hours`        Daily social-media usage
  `gaming_hours`              Daily mobile gaming time
  `work_study_hours`          Daily phone-based work/study activity
  `sleep_hours`               Average sleep duration
  `notifications_per_day`     Daily notification count
  `app_opens_per_day`         Daily application-opening frequency
  `weekend_screen_time`       Weekend smartphone screen time

### Categorical Features

  Feature                  Description
  ------------------------ -------------------------------
  `gender`                 Gender category
  `stress_level`           Reported stress category
  `academic_work_impact`   Academic/work impact category

### Target

`addicted_label` is the binary competition target used for training.


# Notebook 1: `uncover-digital-behavior-smartphone-addiction.ipynb`

## 1. Dataset Introduction

The notebook begins includes exploration of the competition dataset, its includes target variable, and available demographic, lifestyle, academic/work,
and digital-behavior variables.

The initial dataset exploartion includes:

*   Training and testing dimensions.
*   Feature types.
*   Target availability.
*   Missing-data presence.
*   Submission requirements.

## 2. Data Structure and Quality Checks

The notebook performs several initial checks:

*   Dataset shapes.
*   `DataFrame.info()` inspection.
*   Data types.
*   Non-null counts.
*   Missing counts.
*   Summary statistics.
*   Train/test missing-value comparison.
*   Duplicate-row checks.

This creates a baseline understanding of the raw data before transformations are applied.

## 3. Missing-Value Analysis

Missing values are handeled before modeling.

The notebook compares:

*   Missing counts.
*   Missing percentages.
*   Training versus testing missingness.
*   Numerical versus categorical missingness.
*   Missingness before and after processing.

## 4. Behavioral Exploratory Data Analysis

### Age and Screen Time

The notebook includes in-depth exploration of:

*   Daily screen-time distribution.
*   Screen time by age.
*   Maximum screen time by age group.
*   Average screen time by age.
*   Top age groups by screen usage.

### Social Media Usage

The notebook also explores:

*   Social-media usage distribution.
*   Social-media behavior across ages.
*   High social-media-use users.

### Gaming Behavior

Gaming analysis includes:

*   Gaming-hour distribution.
*   Gaming usage by age.
*   High-gaming users.
*   Low-gaming users.
*   Gaming and sleep relationships.
*   Gaming behavior across different age groups.

And it is observed that:

*   Maximum gaming time of **4 hours/day**.
*   Overall average gaming time of approximately **1.01 hours/day**.
*   Highest average gaming usage around ages 26--28.
*   Lowest average gaming usage around age 17.

### Notifications and App Usage

The notebook analyzes:

*   Notification frequency.
*   Notification behavior by age.
*   App-opening frequency.
*   Notification/app-opening relationships.
*   High notification users.
*   High app-opening users.

The notebook provide indepth analysis that an overall average of approximately **133.5 otifications/day**, with age-level averages generally remaining within a relatively narrow range.

### Weekend Screen Time

Weekend screen time is examined against:

*   Age.
*   Daily screen time.
*   Other activity signals.
*   User behavior categories.

### Work and Study

The notebook also explores:

*   Work/study hours.
*   Age-level patterns.
*   Relationships with digital activity.
*   Academic/work-impact categories.

## 5. Extreme-User and High-Activity Analysis

It examines:

*   Maximum daily screen-time users.
*   Maximum social-media users.
*   Maximum gaming users.
*   Maximum notification users.
*   Maximum app-opening users.
*   Maximum weekend-screen-time users.
*  Top 10 users for selected behavioral variables.
*   High screen-time users.

## 6. Behavioral Relationships and Segmentation

The notebook studies behavioral patterns across:

*   Age groups.
*   Gender.
*  Stress levels.
*   Academic/work-impact categories.
*   Screen-time categories.
*   High-activity users.

## 7. Feature Engineering

### Missingness Features

*   Missing-value indicators.
*   `missing_count`

### Time and Activity Features

*   `total_breakdown_hours`
*   `social_ratio`
*   `gaming_ratio`
*   `work_ratio`
*   `unaccounted_screen_time`
*   `screen_to_sleep_ratio`
*   `weekend_vs_daily_ratio`
*   `app_opens_per_hour`
*   `notifications_per_hour`

### Categorical and Binned Features

The notebook additionally creates:

*   Age groups.
*   Re-categorized continuous variables.
*   Floor/round/modulo/fraction-based categorical transformations.
*   Pairwise categorical features interactions such as:

After feature engineering and integration the dataset includes:

*   **691,369 training rows**
*   **296,302 test rows**
*   **107 total model features**
*   **54 categorical features**
*   **53 numerical features**

# Modeling

1. XGBoost
2. LightGBM
3. TabM

* The notebook also trains **TabM**, a neural tabular model available
through `pytabkit`.

* Models Ensembling

# Notebook 2: `predict-smartphone-addiction-ps6e8-v2.ipynb`

* 1. Data Preparation

* 2. Missing-Value Handling

. Feature Engineering 

The engineered numerical features include:

-   `missing_count`
-   `total_screen`
-   `weekend_diff`
-   `social_ratio`
-   `gaming_ratio`
-   `activity_total`
-   `activity_ratio`
-   `screen_sleep_ratio`
-   `sleep_deficit`
-   `engagement`
-   `notif_per_open`
-   `log_notifications`
-   `log_app_opens`
-   `screen_per_open`
-   `screen_x_stress`
-   `social_x_stress`
-   `sleep_x_stress`
-   `gender_freq`
-   `stress_level_freq`
-   `academic_work_impact_freq`

3. RealMLP Model

4. Cross-Validation

5. Training Performance

*   Fold 1 best AUC: **0.96824**
*   Fold 2 reaches approximately **0.96895**
*   Training proceeds for 12 epochs with the best validation epoch
    retained in the prediction pipeline.

7. Prediction and Submission

The final test predictions are craeted with ebst modeling reuslts

# End-to-End Workflow

The overall project has the following pipeline:

```
Competition Dataset
        │
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv
                │
                ▼
        Data Understanding
                │
                ▼
        Data Quality Checks
                │
                ├── Data types
                ├── Missing values
                ├── Duplicates
                └── Summary statistics
                │
                ▼
        Missing-Value Handling
                │
                ▼
        Behavioral EDA
                │
                ├── Screen time
                ├── Social media
                ├── Gaming
                ├── Sleep
                ├── Notifications
                ├── App opens
                ├── Weekend usage
                └── Work/study impact
                │
                ▼
        Feature Engineering
                │
                ├── Ratios
                ├── Interactions
                ├── Missing indicators
                ├── Binning
                ├── Frequency features
                └── Target encoding
                │
                ▼
        Model Training
                │
        ┌───────┴────────┐
        ▼                ▼
  Notebook 1        Notebook 2
        │                │
 XGBoost             RealMLP
 LightGBM               │
 TabM             5-fold × 4 seeds
        │                │
        └───────┬────────┘
                ▼
        Probability Predictions
                │
                ▼
          Model Ensembling
                │
                ▼
       Final Submission File
```

### Evaluation Metric

The project uses **ROC-AUC** as the main evaluation metric.

ROC-AUC measures how well the model ranks positive examples above
negative examples across probability thresholds.

This is appropriate for the competition workflow because the notebooks
generate probability predictions rather than only hard class labels.

The main validation artifacts are:

-   Fold-level AUC.
-   Overall out-of-fold AUC.
-   Multi-seed AUC.
-   Ensemble AUC.