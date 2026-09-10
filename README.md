# AQI Predictor

A beginner-friendly Machine Learning project that predicts the **Air Quality Index (AQI)** using environmental parameters.

The project demonstrates a clear end-to-end workflow, from **raw data preparation and leakage-safe preprocessing to model training, evaluation, and GUI-based prediction**.

## Project Overview

The project predicts AQI using 8 environmental parameters:

* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃
* Temperature
* Humidity

The target variable is **AQI**.

Two regression models are included so learners can compare different approaches:

* **Linear Regression**
* **Decision Tree Regressor** (`max_depth=5`, `random_state=42`)

## Processing Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Train / Test Split
     ↓
Median Imputation
     ↓
Model Training
     ↓
Evaluation
     ↓
AQI Prediction
     ↓
GUI
```

## Data Leakage Prevention

To keep evaluation fair, preprocessing is done with strict train/test separation:

1. The cleaned dataset is split into **80% training** and **20% testing** before imputation.
2. Median values are calculated **only from the training dataset**.
3. Those training medians are applied to fill missing values in the test dataset.
4. Test data is never used to calculate preprocessing statistics.

This helps ensure that model evaluation reflects true generalization performance.

## Files

| File                    | Purpose                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `globalAQI_raw.csv`     | Original/raw AQI dataset                                      |
| `data_split.py`         | Cleans raw data and creates `train_data.csv` / `test_data.csv` |
| `data_process.py`       | Applies training-only median imputation and creates `train_processed.csv` |
| `model_train.py`        | Trains the Linear Regression model                            |
| `Descion_Tree.py`       | Trains the Decision Tree Regressor (`max_depth=5`, `random_state=42`) |
| `evaluate.py`           | Evaluates the Linear Regression model on test data            |
| `evaluate_tree.py`      | Evaluates the Decision Tree model on test data                |
| `check_overfitting.py`  | Reports Decision Tree training performance for overfitting comparison |
| `AQI_model.pkl`         | Saved Linear Regression model                                 |
| `AQI_decision_tree.pkl` | Saved Decision Tree model                                     |
| `interface.py`          | PySide6 GUI for interactive AQI prediction                    |

## Machine Learning

### Models

**Linear Regression** and **Decision Tree Regressor** are both trained on the processed training dataset.

Using both models allows direct comparison between a simple linear method and a non-linear tree-based method.

### Model Evaluation

Both models are evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination

### Overfitting Analysis

Overfitting is checked by comparing Decision Tree training performance (`check_overfitting.py`) with testing performance (`evaluate_tree.py`).

If training metrics are much better than testing metrics, the model may be overfitting. If both are reasonably close, the model is generalizing better.

## GUI

The project includes a graphical interface built using **PySide6**.

The GUI provides:

* Model selection (**Linear Regression** or **Decision Tree**)
* Input fields for all 8 environmental parameters
* AQI prediction output
* AQI category display (GOOD, MODERATE, UNHEALTHY FOR SENSITIVE GROUPS, UNHEALTHY, VERY UNHEALTHY, HAZARDOUS)

## Tools & Libraries

* Python
* Pandas
* Scikit-learn
* Joblib
* PySide6

## How to Run

### 1. Install required libraries

```bash
pip install -r requirements.txt
```

### 2. Create cleaned train/test datasets

```bash
python data_split.py
```

This creates:

```text
train_data.csv
test_data.csv
```

### 3. Apply training-set median imputation

```bash
python data_process.py
```

This creates:

```text
train_processed.csv
```

### 4. Train models

```bash
python model_train.py
python Descion_Tree.py
```

This creates/updates:

```text
AQI_model.pkl
AQI_decision_tree.pkl
```

### 5. Evaluate models

```bash
python evaluate.py
python evaluate_tree.py
```

Both scripts print MAE, RMSE, and R² on the test set.

### 6. Check Decision Tree overfitting behavior

```bash
python check_overfitting.py
```

Compare this training output with `evaluate_tree.py` test output to assess generalization.

### 7. Run the GUI

```bash
python interface.py
```

## Project Workflow

```text
globalAQI_raw.csv
        ↓
data_split.py
        ↓
train_data.csv + test_data.csv
        ↓
data_process.py
        ↓
train_processed.csv
        ↓
model_train.py + Descion_Tree.py
        ↓
AQI_model.pkl + AQI_decision_tree.pkl
        ↓
evaluate.py + evaluate_tree.py + check_overfitting.py
        ↓
interface.py
        ↓
AQI Prediction
```

## Note

This project is developed for **educational and learning purposes** to demonstrate the Machine Learning workflow of preprocessing, training, evaluation, and GUI integration.

It is **not a professional or fully real-world accurate AQI prediction system**. Predictions are intended for learning and demonstration only, and should not be treated as authoritative air quality measurements.
