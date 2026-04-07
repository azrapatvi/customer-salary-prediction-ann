# 💎 Salary Oracle — Customer Salary Prediction with ANN

> A deep learning web app that predicts a bank customer's estimated salary using an Artificial Neural Network (ANN), built with TensorFlow/Keras and deployed via a sleek Streamlit interface.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline (Notebook Walkthrough)](#-ml-pipeline-notebook-walkthrough)
  - [1. Data Loading & Exploration](#1-data-loading--exploration)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering & Encoding](#3-feature-engineering--encoding)
  - [4. Preprocessing Pipeline](#4-preprocessing-pipeline)
  - [5. Train/Test Split](#5-traintest-split)
  - [6. ANN Model Architecture](#6-ann-model-architecture)
  - [7. Training with Callbacks](#7-training-with-callbacks)
  - [8. Evaluation](#8-evaluation)
  - [9. Saving Artifacts](#9-saving-artifacts)
- [Streamlit App (main.py)](#-streamlit-app-mainpy)
  - [App Layout](#app-layout)
  - [Input Fields](#input-fields)
  - [Prediction Flow](#prediction-flow)
- [How to Run Locally](#-how-to-run-locally)
- [Models Folder](#-models-folder)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

---

## 🧠 Project Overview

**Salary Oracle** is an end-to-end machine learning project that:

1. Trains an **Artificial Neural Network (ANN)** regression model on a bank customer dataset to predict the `EstimatedSalary` of a customer.
2. Saves the trained model and preprocessing artifacts.
3. Serves the model through a **Streamlit web app** (`main.py`) with a modern, eye-catching dark-themed UI.

The model takes 9 customer attributes as input — such as credit score, geography, gender, age, tenure, and account balance — and outputs the predicted annual salary in Euros.

---

## 🚀 Live Demo

> Run locally using the instructions below. No hosted demo at this time.

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x / Keras |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) |
| Model Persistence | Joblib, Keras `.keras` format, JSON |
| Web App | Streamlit |
| Visualization | Seaborn, Matplotlib, TensorBoard |

---

## 📁 Project Structure

```
salary-oracle/
│
├── main.py                        # Streamlit web application
├── salary_prediction.ipynb        # Full ML pipeline notebook
│
├── models/
│   ├── model.keras                # Trained ANN model
│   ├── preprocessor.joblib        # Fitted ColumnTransformer (scaler + encoder)
│   └── gender_encoding.json       # Label encoding map for Gender column
│
├── regressionlogs/                # TensorBoard training logs
│   └── fit/
│       └── <timestamp>/
│
├── requirements.txt               # Python dependencies
└── README.md                      # You are here
```

---

## 📊 Dataset

- **Source:** [Churn Modelling Dataset](https://raw.githubusercontent.com/krishnaik06/ANN-CLassification-Churn/refs/heads/main/Churn_Modelling.csv)
- **Rows:** 10,000 customer records
- **Original Columns:** `RowNumber`, `CustomerId`, `Surname`, `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`
- **Target Variable:** `EstimatedSalary` (continuous, regression task)

---

## 🔬 ML Pipeline (Notebook Walkthrough)

The notebook `salary_prediction.ipynb` contains the complete end-to-end pipeline. Here's a detailed explanation of each step:

---

### 1. Data Loading & Exploration

```python
df = pd.read_csv('https://raw.githubusercontent.com/..../Churn_Modelling.csv')
df.head()
df.isnull().sum()
df.shape
df.columns
df.info()
```

- The dataset is loaded directly from a public GitHub URL using `pd.read_csv()`.
- Basic EDA checks are performed: shape, column names, null values, and data types.
- `df.info()` confirms there are no missing values and reveals which columns are categorical (`object` dtype) vs. numerical.

---

### 2. Data Cleaning

```python
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.duplicated().sum()
df.columns = df.columns.str.lower()
```

- **Dropped columns:** `RowNumber`, `CustomerId`, and `Surname` are identifiers that carry no predictive signal for salary estimation.
- **Duplicate check:** Confirms there are no duplicate rows in the dataset.
- **Column normalization:** All column names are converted to lowercase to keep the pipeline consistent and avoid case-sensitivity bugs.

---

### 3. Feature Engineering & Encoding

```python
cat_cols = [i for i in df.columns if df[i].dtypes == 'O']
num_cols = [i for i in df.columns if df[i].dtypes != 'O']

df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
```

- Categorical and numerical columns are identified dynamically using dtype inspection.
- **Gender encoding:** `Gender` is binary label-encoded manually (`Female → 1`, `Male → 0`). This mapping is saved as a JSON file so the Streamlit app can apply the same encoding at inference time.
- After encoding, `Gender` moves from `cat_cols` to `num_cols`.
- The remaining categorical column is `Geography` (France, Germany, Spain), which is handled by `OneHotEncoder` in the preprocessing pipeline.

---

### 4. Preprocessing Pipeline

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

scaler = StandardScaler()
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ("OneHotEncoder", ohe, cat_cols),
    ("StandardScaler", scaler, num_cols)
])
```

- A `ColumnTransformer` is built to apply two different transformations simultaneously:
  - **OneHotEncoder** on `cat_cols` (i.e., `geography`): Converts categorical geography values into binary dummy variables. `drop='first'` avoids multicollinearity by dropping one category (dummy variable trap). `handle_unknown='ignore'` ensures graceful inference if an unseen geography value is passed.
  - **StandardScaler** on `num_cols` (all remaining numeric features): Normalizes values to zero mean and unit variance, which is critical for neural network convergence.
- The target variable `estimatedsalary` and `exited` column are excluded from `num_cols` before preprocessing.

---

### 5. Train/Test Split

```python
X = df.drop(['estimatedsalary', 'exited'], axis=1)
y = df['estimatedsalary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled  = preprocessor.transform(X_test)
```

- Features (`X`) are all columns except the target `estimatedsalary` and `exited` (which is a churn label, not relevant to salary prediction).
- An **80/20 train-test split** is used with `random_state=42` for reproducibility.
- The preprocessor is **fit only on training data** (`fit_transform`) and then applied to test data (`transform`) — this is the correct approach to prevent data leakage.

---

### 6. ANN Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
```

The neural network is a simple 3-layer feedforward ANN:

| Layer | Units | Activation | Purpose |
|---|---|---|---|
| Input + Dense | 64 | ReLU | Learns complex non-linear patterns |
| Hidden Dense | 32 | ReLU | Further feature abstraction |
| Output Dense | 1 | Linear (none) | Regression output — raw salary value |

- **Optimizer:** `Adam` — adaptive learning rate optimizer, ideal for this problem size.
- **Loss:** `Mean Absolute Error (MAE)` — chosen because it's robust to outliers in salary data and interpretable (in currency units).
- **Metric:** `MAE` is tracked during training to monitor real-world prediction error.

---

### 7. Training with Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=200,
    callbacks=[tensorboard_callback, es]
)
```

- **EarlyStopping:** Monitors `val_loss` and stops training if it doesn't improve for 10 consecutive epochs. `restore_best_weights=True` ensures the best checkpoint is preserved, preventing overfitting.
- **TensorBoard:** Logs training metrics (loss, MAE, weight histograms) to a timestamped directory under `regressionlogs/`. You can visualize live training with `%tensorboard --logdir regressionlogs/`.
- **Max epochs:** Set to 200, but early stopping will typically trigger much earlier.

---

### 8. Evaluation

```python
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print("Test MAE:", test_mae)
```

- After training, the model is evaluated on the held-out test set.
- **Test MAE** gives the average absolute error in salary prediction (in Euros). Lower is better.

---

### 9. Saving Artifacts

```python
import joblib, json

joblib.dump(preprocessor, "models/preprocessor.joblib", compress=3)

with open('models/gender_encoding.json', 'w') as f:
    json.dump(gender_encoding, f)

model.save('models/model.keras')
```

Three artifacts are saved to the `models/` directory:

| File | Purpose |
|---|---|
| `model.keras` | The trained ANN (TensorFlow native format) |
| `preprocessor.joblib` | The fitted `ColumnTransformer` (scaler + encoder) |
| `gender_encoding.json` | The `{"Female": 1, "Male": 0}` mapping dict |

All three are required at inference time by `main.py`.

---

## 🖥 Streamlit App (`main.py`)

### App Layout

The app features a fully custom dark-themed UI built with injected CSS, Google Fonts (`Syne` + `DM Sans`), and animated components:

- **Hero Banner** — Gradient header with tagline and glowing orb effects
- **3-Column Input Grid** — All 9 customer fields organized in a clean card layout
- **Live Summary Chips** — Real-time summary of all entered values shown below the form
- **Collapsible Data Preview** — Raw DataFrame preview hidden in an expander for cleanliness
- **Animated Predict Button** — Glowing purple CTA button with hover lift effect
- **Result Box** — Animated salary result displayed with gradient typography

### Input Fields

| Field | Type | Range / Options |
|---|---|---|
| Credit Score | Number input | 300 – 900 |
| Geography | Dropdown | France, Germany, Spain |
| Gender | Dropdown | Male, Female |
| Age | Number input | 18 – 100 |
| Tenure | Number input | 0 – 10 years |
| Balance | Float input | ≥ 0.00 |
| No. of Products | Number input | 1 – 4 |
| Has Credit Card | Dropdown | Yes / No |
| Active Member | Dropdown | Yes / No |

### Prediction Flow

```
User fills inputs
        ↓
"Run Prediction" clicked
        ↓
Load gender_encoding.json  →  map Gender column
        ↓
Load preprocessor.joblib   →  transform input DataFrame
        ↓
Load model.keras           →  model.predict(X_processed)
        ↓
Display predicted salary in animated result box
```

The app loads the model files fresh on every prediction click (not on app startup), keeping memory usage clean and making model swapping straightforward.

---

## ⚙️ How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/salary-oracle.git
cd salary-oracle
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (or use pre-trained artifacts)

Open and run all cells in `salary_prediction.ipynb`. This will generate the `models/` folder with all three required files.

> ⚠️ If you skip this step, make sure the `models/` directory already contains `model.keras`, `preprocessor.joblib`, and `gender_encoding.json`.

### 5. Launch the Streamlit app

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`.

---

## 🗂 Models Folder


```
models/
├── model.keras               # ~500KB typical size
├── preprocessor.joblib       # Fitted sklearn pipeline
└── gender_encoding.json      # {"Female": 1, "Male": 0}
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Loss Function | Mean Absolute Error (MAE) |
| Training | Up to 200 epochs with early stopping |
| Typical Test MAE | ~eur 50,000 – 60,000 (salary scale dependent) |

> Note: The `EstimatedSalary` feature in the Churn dataset is essentially **randomly distributed** between customers (it is synthetically generated), so MAE will remain relatively high regardless of model complexity. This project is primarily a demonstration of the end-to-end ANN regression pipeline.

---

## 🔮 Future Improvements

- [ ] Add confidence intervals or uncertainty estimation to the prediction
- [ ] Experiment with deeper architectures and dropout regularization
- [ ] Add SHAP explainability to show which features drive each prediction
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Add batch prediction — upload a CSV and predict for multiple customers at once
- [ ] Include a TensorBoard visualization panel inside the Streamlit app

---

## 👤 Author

Built with ❤️ using TensorFlow, Scikit-learn, and Streamlit.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
