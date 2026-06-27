# Student Marks Prediction Analysis

An end-to-end data science project demonstrating exploratory data analysis (EDA), data visualization, and predictive modeling using regression techniques on a student performance dataset. The project evaluates multiple machine learning algorithms to predict student marks based on their study habits and course load.

---

## 📌 Project Overview
This repository contains a comprehensive analysis of factors influencing students' academic performance, specifically their final marks. Using Python's powerful data science library stack (`pandas`, `numpy`, `seaborn`, `matplotlib`, and `scikit-learn`), we conduct a full lifecycle implementation: from data ingestion and cleaning to advanced regression modeling using regularized polynomial transformations.

### Key Objectives:
* Perform exploratory data exploration to understand feature distributions and correlations.
* Visualize relationships between independent variables (study time, number of courses) and the target variable (marks).
* Implement and train multiple regression models including **Multiple Linear Regression**, **Polynomial Regression (Degree 2)**, and **Ridge Regression**.
* Evaluate models using robust validation techinques ($5$-Fold Cross-Validation) and statistical metrics ($R^2$, MAE, RMSE).

---

## 📊 Dataset Description
The dataset used for this project is stored in `Student_Marks.csv` and comprises 100 observations of student data with no missing or null values. 

### Features:
1. **`number_courses`** *(Integer)*: The total number of courses the student has opted for (ranging from 3 to 8).
2. **`time_study`** *(Float)*: The average number of hours spent studying per day.
3. **`Marks`** *(Float - Target)*: The final marks scored by the student.

### Statistical Summary:
| Metric | `number_courses` | `time_study` | `Marks` |
| :--- | :---: | :---: | :---: |
| **Count** | 100.0000 | 100.0000 | 100.0000 |
| **Mean** | 5.2900 | 4.0771 | 24.4177 |
| **Std Dev** | 1.7995 | 2.3729 | 14.3262 |
| **Min** | 3.0000 | 0.0960 | 5.6090 |
| **25%** | 4.0000 | 2.0585 | 12.6330 |
| **50%** | 5.0000 | 4.0220 | 20.0595 |
| **75%** | 7.0000 | 6.1793 | 36.6763 |
| **Max** | 8.0000 | 7.9570 | 55.2990 |

---

## 📈 Exploratory Data Analysis & Insights
Several visualizations were generated during the analysis to decode underlying trends:
* **Feature Distributions:** Distribution plots showed a uniform spread of study hours, spanning from minimal daily study to nearly 8 hours per day.
* **Scatter & Trend Plots:** A distinct non-linear, slightly quadratic relationship was identified between `time_study` and `Marks`, indicating that as study time increases, academic performance rises exponentially rather than strictly linearly.
* **Correlation Matrix:** A correlation analysis revealed an exceptionally strong positive correlation between study hours and marks, and a moderate correlation between the number of courses and marks.

---

## 🤖 Model Development & Evaluation

The dataset was split into an **80% training set** and a **20% testing set** to ensure fair evaluation. Due to the observed non-linear behavior in the EDA phase, polynomial features were introduced.

### Performance Metrics Table

| Model Name | Test $R^2$ Score | Cross-Val $R^2$ (Mean) | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** | 0.9460 | 0.9325 | 3.0793 | 3.7684 |
| **Polynomial Regression (Degree 2)** | 0.9997 | 0.9996 | 0.2278 | 0.2853 |
| **Ridge Regression (Regularized Poly)** | 0.9996 | 0.9995 | 0.2410 | 0.3066 |

### Key Findings:
1. **Linearity vs. Non-Linearity:** The baseline Multiple Linear Regression model performed highly adequately ($R^2 = 0.9460$). However, it suffered from a significantly higher root-mean-squared error (RMSE = 3.7684), indicating structural underfitting due to the non-linear relationship.
2. **Polynomial Excellence:** Introducing 2nd-degree polynomial combinations of the features boosted the explanation of variance to an exceptional **99.97%**. 
3. **Generalization Overfitting Check:** The 5-Fold Cross-Validation scores match the test scores almost perfectly across all candidates, verifying that the models generalize excellently and are completely stable across different data folds.
4. **Final Model Choice:** The **Ridge Regression model with Polynomial Features** was selected as the final production model due to its added regularized stability ($ lpha = 1.0$), which natively reduces risks of over-parameterization while maintaining an outstanding $R^2$ of **0.9996** and an extremely low MAE of **0.2410**.

---

## 🛠️ Environment & Setup

### Prerequisites
Make sure you have Python 3.8+ installed along with the following data science tools:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Project
1. Clone this repository or download the source files.
2. Ensure `Student_Marks.csv` is located in the same working directory as your execution script.
3. Run the Jupyter Notebook or Python file to view output logs, generated correlation heatmaps, and regression plots:
```bash
jupyter notebook Student_Marks_Analysis.ipynb
```

---

## 👥 Author
* **Vishal M.**
* Computer Science Student, Chennai Institute of Technology

---
