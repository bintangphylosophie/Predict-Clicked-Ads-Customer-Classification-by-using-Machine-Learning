# Predict Clicked Ads Customer Classification by using Machine Learning

<br>
<p align="center">
<img src="https://github.com/user-attachments/assets/18d9a370-0ae4-4033-a052-56ca3a4eaf60" width="500">
</p>
<br>
This dataset appears to be about user behavior on a website, likely to explore how demographic and behavioral factors impact the likelihood of clicking on an ad. This dataset is from a fictional company from January to July in 2016. 

## 📚 Installation

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Sklearn

If you don't have Python installed yet, it's recommended that you install [the Anaconda distribution](https://www.anaconda.com/distribution/) of Python, which already has the above packages and more included.

To install the Python libraries, you can use pip:

```bash
pip install numpy pandas matplotlib seaborn scipy sklearn imbalanced-learn
```
To run the Jupyter notebook, you also need to have Jupyter installed. If you installed Python using Anaconda, you already have Jupyter installed. If not, you can install it using pip:
```bash
pip install jupyter
```

Once you have Python and the necessary libraries, you can run the project using Jupyter Notebook:
```bash
jupyter notebook Predict Clicked Ads Customer Classification by using Machine Learning.ipynb

```
## Project Overview
A company in Indonesia wants to know the effectiveness of the advertisement they broadcast. This is important for the company to know how successful the advertisement being marketed is so that it can attract customers to see the advertisement.
By processing historical advertisement data and finding insights and patterns that occur, it can help companies determine marketing targets. The focus of this case is to create a machine learning classification model that functions to determine the right target customers.

## Problem
The company wants to evaluate the effectiveness of its advertisements in attracting customer engagement. Understanding which customers are more likely to click on ads will help improve targeted marketing efforts and optimize advertising spend. However, the company currently lacks a systematic approach to identifying potential customers who are most likely to interact with their ads.

## 🎯 Goals 
The goal of this project is to develop a machine learning classification model to predict whether a customer will click on an ad based on their demographic and behavioral data. By analyzing historical advertisement data, we aim to:
- Optimizing ad spend by targeting relevant users.
- Increasing ad engagement and conversion rates.
- Segmenting customers based on their behavior and demographics.
## 🏁 Obective 
1. Identify key factors that influence ad clicks, such as time spent on the site, internet usage, age, income, and location.
2. Develop a predictive model to classify users into those who are likely to click on ads and those who are not.
3. Provide insights that can help the company refine its marketing strategies and improve ad targeting for better engagement.

The process will go through the following steps to achieve the objectives:
1. Data Understanding
2. Data Preprocessing
3. Feature Engineering
4. Insight
5. Exploratory Data Analysis
6. Data Preparation
7. Machine Learning
<br>
<br>

# 🔎Stage 1: Data Understanding and Preprocessing 

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/212749726-d36b8253-74bb-4509-870d-e29ed3b8ff4a.gif" width="500">
</p>
<br>

## 📊 About Dataset 

### 📋Dataset consist :
- There are 1000 rows and 11 columns in the dataset. 
- The target column in this dataset is `Clicked on Ad`.
- The dataset contains several missing values, which are filled considering the number of missing values, their skewness, and the data type
- The dataset does not have duplicated data
- The dataset does not have outliers
- Several inconsistent data values have been tidied up.


### 📝Features

| Feature | Explanation |
|---------|-------------|
| Daily Time Spent on Site | The amount of time (in minutes) a user spends on the website daily. |
| Age | The user’s age. |
| Area Income | The average income of the user's geographical area. |
| Daily Internet Usage | The amount of time (in minutes) spent online daily. |
| Male | A binary indicator (1 = Male, 0 = Female). |
| Timestamp | The exact time when the user interacted with the ad. |
| Clicked on Ad | A binary label (1 = Clicked, 0 = Not Clicked). |
| City | The city where the user resides. |
| Province | The province/state of the user. |
| Category | The category of the ad displayed to the user. |
<br>
<br>



# ⚙️ Stage 2: Feature Engineering

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/221352995-5ac18bdf-1a19-4f99-bbb6-77559b220470.gif" width="400">
</p>


This is the second stage of  focusing on feature engineering of the dataset. The main goal of this stage is to clean and transform the raw data to make it suitable for data analysis. 🧹🔄

<br>

**Key steps in this stage include:**
## 1. Column: TimeStamp
- `Timestamp_month`
- `Timestamp_day`
- `HourOfDay`
## 2. Column: Age
- `AgeCategory`
## 3. Column: TimeSpentToInternetUsage

<br>
<br>

# 🚀 Stage 3: Insight

<br>
<p align="center">
<img src="https://media0.giphy.com/media/Y4PkFXkfTeEKqGBBsC/giphy.gif?cid=ecf05e47numetagrmbl2rxf6x0lpcjgq1s340dfdh1oi0x9w&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="420">
</p>

This is the next phase of the project, focusing on gaining insights. Here are some valuable insights derived from the dataset
<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/f8b2d984-cba0-4f8c-89dd-f586cc51e671" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/51079050-180f-4caa-871f-55919f9c65eb" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/f7b88c48-f946-4c68-8fae-35b9b5282bb1" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/e7fc90c3-d9fc-4728-853c-5b978929f834" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/58d88885-0519-4862-afc7-5f907252d400" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/e3dd77ed-388f-4cc3-b829-03a4f7aa83bd" width="300"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/d1098e4c-a6d0-4f87-8a07-19644c0d5c89" width="300"></td>
      <td><img src="https://github.com/user-attachments/assets/a0ec9fec-0d60-45c1-9f8c-1a31ba5e53ad" width="300"></td>
    </tr>
  </table>
</div>



# ⚙️ Stage 4: Data Preprocessing
This stage is focusing on data preprocessing of the dataset transform the data to make it suitable for modeling
## 1. **Feature Encoding** 🏷️<br>
I encode all of our categorical features (strings) using the label encoding method. All features have been encoded at the feature extraction stage, given that our features are ordinal data and the majority of machine learning algorithms perform better with numerical data.

## 2. **Data Scaling** 🏷️<br>
Data scaling is the process of transforming feature values within a dataset to ensure they have a uniform range. I performed using MinMaxScaler to improve machine learning algorithm performance that will be perform in next stage.

## 3. **Feature Selection** 🎯<br>
Since we are working with categorical features and a classification problem, SelectKBest with chi2 is a great choice because it is: fast, helps remove irrelevant features, works well with encoded categorical data. This method is commonly used to reduce dimensionality by keeping only the most relevant features for predictive modeling. We only keep top 5 features.

## 4. **Data Splitting**<br>
Data splitting is the process of dividing a dataset into different subsets to train, validate, and test a machine learning model. 
The main goal of this stage is to build and evaluate models that can predict the target variable based on the preprocessed data. The data splitted into 20% train and 80% test.


Prior to this, the dataset was split into training and testing sets. After completing all preprocessing steps, the data is clean and ready for machine learning to predict the target.🤖

<br>
<br>


# 🚀 Stage 5: Modelling
<br>
<p align="center">
<img src="https://media0.giphy.com/media/Y4PkFXkfTeEKqGBBsC/giphy.gif?cid=ecf05e47numetagrmbl2rxf6x0lpcjgq1s340dfdh1oi0x9w&ep=v1_gifs_related&rid=giphy.gif&ct=g" width="420">
</p>
The main goal of this stage is to build and evaluate models that can predict the target variable based on the preprocessed data. 🎯

## 📚 Installation

This project requires Python and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Sklearn
- shap

If you don't have Python installed yet, it's recommended that you install [the Anaconda distribution](https://www.anaconda.com/distribution/) of Python, which already has the above packages and more included.

To install the Python libraries, you can use pip:

```bash
pip install numpy pandas matplotlib seaborn scipy sklearn imbalanced-learn xgboost shap
```
To run the Jupyter notebook, you also need to have Jupyter installed. If you installed Python using Anaconda, you already have Jupyter installed. If not, you can install it using pip:
```bash
pip install jupyter
```

Once you have Python and the necessary libraries, you can run the project using Jupyter Notebook:
```bash
jupyter notebook Modelling_Merged.ipynb
```


**Key steps in this last stage include:**

## 1. **Models**: 🏗️<br>

I experimented with other algorithms. A total of 5 algorithms were tested during the experiment, including:
- **Support Vector Machine**
- **Gradient Boosting**
- **Decision Tree**
- **Random Forest**
- **Logistic Regression**

<br> 

**Tuning:** Hyperparameter tuning was performed only on the 1 best algorithms (those with the lowest MAE).

## 2. **Model Training and Evaluation**: 🏋️‍♀️🎯<br>

The following are the prediction results with the highest Accurcy and ROC-AUC:

### Model Results
| Model Name | Accuracy | ROC AUC | 
|------------|--------------|-------------|
| Support Vector Machine | 0.67 | 0.50 | 
| Gradient Boosting | 0.94 | 0.94 | 
| Decision Tree | 0.93 | 0.93 | 
| Random Forest | 0.89 | 0.85 | 
| Logistic Regression | 0.70 | 0.57 | 


We discovered that the Gradient Boosting Model with the highest Accuracy (0.94) and ROC AUC (0.94), stability compared to other models. 

## 3. **Model Selection**: 🥇<br>

### Model Results
| Model Name | Accuracy | ROC AUC | 
|------------|--------------|-------------|
| Gradient Boosting | **0.94** | **0.94** |

with confussion matrich and ROC AUC score like this:

<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/259784b2-2b39-4912-b555-26b4d56dfdeb" width="400"></td>
      <td><img src="https://github.com/user-attachments/assets/ffb76db3-751f-4ef9-b82a-56c05b3ec943" width="400"></td>
    </tr>
  </table>
</div>

## 🔑 Feature Importance 
Based on the Gradient Boosting model Feature Importance:

<div align="center">
  <img src="https://github.com/user-attachments/assets/200d233e-1cdc-403e-bb6a-8887b523f375" width="400">
</div>

<br>

The top importance feature with score > 0.01 are :
- `AlasanResign`
- `AsalDaerah_JakartaSelatan`
- `UsiaKaryawan`
- `HiringPlatform_Diversity_Jobfair`


## ✅ Business Recommendation
To improve retention, the company deploys a machine learning model to identify key factors leading to resignations. with AI-driven insights, the company designs targeted retention strategies:

### The model strongly relies on "AlasanResign" (Resignation Reason), meaning that resignation trends must be analyzed deeply to take proactive action. Since working hour is the dominant factor, some action recommend:
- **Analyze Overtime Trends:** Check which departments consistently work overtime and why.
- **Survey Employees:** Get feedback on workload, stress levels, and preferred work hours.
- **Compare Productivity Metrics:** Measure performance vs. hours worked to find the optimal balance.
- **Implement Flexible Working Hours:** Allow employees to choose a start time (e.g., 7 AM - 3 PM, 9 AM - 5 PM, 11 AM - 7 PM)
- **Implement Hybrid & Remote Work Options:** Allow employees to work 2-3 days from home per week.
- **Reduce Unnecessary Meetings & Improve Time Efficiency:** Limit Meetings to 30-45 Minutes, Set "No-Meeting Days“,Block 1-2 days per week for deep work without interruptions.

### The model detects that employees from Jakarta Selatan and mid-employees have a higher risk of resigning. 
- **Personalized Exit Interviews:** Focus on  mid-employee and Jakarta Selatan employees to understand their concerns, "Stay Interviews": Monthly check-ins with at-risk employees
- **Internal Mentorship Programs:** Pair mid-level employees with senior mentors.

#### Machine learning provided valuable insights, but human intervention was key in designing effective retention strategies. XYZ Corp launches AI-powered retention programs for 6 months and tracks the impact.

## 📝 Business Impact Simulation  
Company Name Profile: **XYZ Corp**
A tech company, XYZ Corp, has been struggling with high employee turnover, affecting productivity, morale, and recruitment costs. To improve retention, the company deploys a machine learning model to identify key factors leading to resignations. with AI-driven insights, the company designs targeted retention strategies.

<div align="center">
  <img src="https://github.com/user-attachments/assets/78c8ecef-dbf8-444e-9c8c-4906da6ebf09" width="600">
</div>


## Acknowledgements🌟

I would like to express our deepest appreciation to Rakamin Academy for providing the opportunity to work on this exciting project. The experience and knowledge we gained throughout this journey have been invaluable.

Finally, I would like to thank those who provided their support and encouragement throughout my journey.

Regards, Bintang Phylosophie

<br>
<p align="center">
<img src="https://media1.giphy.com/media/3ohs7JG6cq7EWesFcQ/giphy.gif?cid=ecf05e47v1a2kre6ziee4vjvkc67vxhrwh8ho4089wc0aqli&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="800">
</p>


