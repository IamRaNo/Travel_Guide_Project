# 📍 Travel Recommendation System

A **Machine Learning model** to suggest travel destinations in India based on user preferences.

---

## 🚀 Concept

This project takes user input on preferences like **zone**, **type of place**, **best time to visit**, **entrance fee**, and other features — and recommends the most relevant places to explore in India.

---

## 📂 Dataset

- **Source:** Collected from Kaggle  
- **Size:** 325 rows × 16 columns  
- **Features:**  
  - Initially: 5 numeric + 11 categorical columns  
  - After cleaning: Refined and categorized for better model usability  
- **Coverage:** Places spread across all regions of India

---

## 🔍 Data Highlights

- No duplicate values except 2 similar rows with only entrance fee differences (removed)
- 90% nulls in `weekly off` column (dropped)
- No column with a single constant value
- Target column (`name`) has 321 unique values
- One unnecessary index column (`Unnamed: 0`) removed

---

## 🧹 Data Cleaning & Preparation

### 🔑 Key preprocessing steps

- Dropped `Unnamed: 0` and other unnecessary columns (`significance`, `weekly off`, `rating`, `number of reviews`)
- Renamed columns for better clarity and converted names to lowercase
- Converted `time to visit` from hours to minutes, then binned into categories
- Grouped `number of reviews` into categories for `footfall` estimation
- Standardized duplicate place names by adding city suffixes
- Simplified `type` and `established` columns by merging similar categories
- Cleaned `entrance fee` values into categories
- Merged sparse classes in `best time` column
- Dropped the `city` column to reduce unnecessary granularity
- Added a `footfall` feature
- Ensured all final columns are relevant and consistent for modeling

---

## 🛠️ Tools & Libraries

- **Python**
- **Pandas**, **NumPy** – Data handling and preprocessing
- **Scikit-learn** – OneHotEncoder, ColumnTransformer, Nearest Neighbors
- **Joblib** – Model and data serialization
- **Jupyter Notebook**, **VS Code**, **PyCharm** – Development environment
- **GitHub** – Version control
- **Kaggle** – Data collection source

---

## 🧩 Model

### ⚙️ Preprocessing

One-hot encoding for categorical columns using `ColumnTransformer`.

### 🤖 Algorithm

`NearestNeighbors` with cosine similarity to find similar places based on user input.

### 🔄 Workflow

1. User provides input (zone, type, fee, best time, etc.)
2. Filter data by selected zone
3. Encode features using the saved preprocessor
4. Find nearest neighbors within the filtered data
5. Recommend top matches to the user

---
