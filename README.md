# AQI Predictor

A beginner-friendly Machine Learning project that predicts the **Air Quality Index (AQI)** using environmental parameters.

The project demonstrates a simple end-to-end Machine Learning workflow, from **data preprocessing and model training to evaluation and GUI-based prediction**.

## Project Overview

The project takes environmental measurements as input and uses a **Linear Regression** model to predict AQI.

The project includes:

* Data preprocessing
* Missing-value handling using **Median Imputation**
* Linear Regression model
* Model evaluation
* Graphical User Interface (GUI)

## Environmental Parameters

The model uses 8 input parameters:

* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃
* Temperature
* Humidity

The target variable is **AQI**.

## Processing Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Median Imputation
     ↓
Clean Dataset
     ↓
Train / Test Split
     ↓
Linear Regression
     ↓
Model Evaluation
     ↓
AQI Prediction
     ↓
GUI
```

## Files

| File                | Purpose                                |
| ------------------- | -------------------------------------- |
| `globalAQI_raw.csv` | Original/raw dataset                   |
| `data_process.py`   | Cleans and preprocesses the dataset    |
| `clean_data.csv`    | Processed dataset                      |
| `model_train.py`     | Trains the Linear Regression model     |
| `AQI_model.pkl`     | Saved trained model                    |
| `evaluate.py`       | Evaluates model performance            |
| `interface.py`      | Graphical interface for AQI prediction |
| `README.md`         | Project documentation                  |

## Machine Learning

### Algorithm

**Linear Regression**

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

The split uses `random_state=42`.

### Model Evaluation

The model is evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination

## Data Preprocessing

The preprocessing script:

1. Selects the required parameters.
2. Converts values to numeric format.
3. Handles invalid pollution values.
4. Checks humidity limits.
5. Removes duplicate rows.
6. Uses **median imputation** for missing input values.
7. Removes rows where AQI is missing.
8. Saves the processed dataset as `clean_data.csv`.

## GUI

The project includes a graphical interface built using **PySide6**.

Users can enter the 8 environmental parameters and receive an AQI prediction.

The GUI also displays an AQI category such as:

* GOOD
* MODERATE
* UNHEALTHY FOR SENSITIVE GROUPS
* UNHEALTHY
* VERY UNHEALTHY
* HAZARDOUS

## Tools & Libraries

* Python
* Pandas
* Scikit-learn
* Joblib
* PySide6

## How to Run

### 1. Install the required libraries

```bash
pip install pandas scikit-learn joblib PySide6
```

### 2. Process the dataset

```bash
python data_process.py
```

This creates:

```text
clean_data.csv
```

### 3. Train the model

```bash
python model_train.py
```

This creates:

```text
AQI_model.pkl
```

### 4. Evaluate the model

```bash
python evaluate.py
```

This displays the model's MAE, RMSE and R² score.

### 5. Run the GUI

```bash
python interface.py
```

## Project Workflow

```text
globalAQI_raw.csv
        ↓
data_process.py
        ↓
clean_data.csv
        ↓
model_train.py
        ↓
AQI_model.pkl
        ↓
interface.py
        ↓
AQI Prediction
```

## Note

This project is developed for **educational and learning purposes** to demonstrate the basic Machine Learning workflow of data preprocessing, model training, evaluation, and GUI integration.

It is **not a professional or fully real-world accurate AQI prediction system**. The predictions are intended for demonstration and learning purposes and should not be treated as an authoritative measurement of actual air quality.
