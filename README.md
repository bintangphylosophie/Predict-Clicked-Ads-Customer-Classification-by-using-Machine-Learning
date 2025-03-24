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
jupyter notebook Predict Clicked Ads Customer Classification by Using Machine Learning.ipynb
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

**Cross Validation:** 

## 2. **Model Training and Evaluation**: 🏋️‍♀️🎯<br>

The following are the prediction results with the highest Precision score:

### Model Results
| Model Name | Precision | CV | 
|------------|--------------|-------------|
| Support Vector Machine | 0.96 | 0.98 | 
| Gradient Boosting | 0.94 | 0.96 | 
| Decision Tree | 0.91 | 0.96 | 
| Random Forest | 0.94 | 0.96 | 
| Logistic Regression | 0.95 | 0.94 | 


We discovered that the **Support Vector Machine (SVM) Model** with the highest Precision (0.96) and after Cross Validation (0.98), stability compared to other models. 

## 3. **Model Selection**: 🥇<br>

### Model Results
| Model Name | Accuracy | ROC AUC | 
|------------|--------------|-------------|
| Support Vector Machine (SVM) | **0.96** | **0.98** |

with confussion matric and ROC AUC curve like this:

<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/15c43ded-71c0-4793-8497-8e04f2f1efc0" width="400"></td>
      <td><img src="https://github.com/user-attachments/assets/c4fb754b-bb11-40cb-a5e5-eec1241ef45c" width="400"></td>
    </tr>
  </table>
</div>

## 🔑 Feature Importance 
Based on the Gradient Boosting model Feature Importance:

<div align="center">
  <img src="https://github.com/user-attachments/assets/200d233e-1cdc-403e-bb6a-8887b523f375](https://github.com/user-attachments/assets/632748d1-e5e0-4f97-a358-c90ef8d9c399" width="400">
</div>

<br>
Support Vector Machines (SVM) do not inherently provide feature importance scores like tree-based models do. However, here are some alternative ways to assess feature importance in SVM. In this case, we get the highest Precision score useing SVMs with non-linear kernels (RBF), so permutation importance will be useful.
The top 3 importance feature with higest are :
- `Daily Internet Usage`
- `Daily Time Spent on Site`
- `Age`
Those 3 features significantly influence our model, indicating that we can focus on using those 3 features in understanding user behavior and optimizing ad campaigns.


## ✅ Business Recommendation
To optimizing ad spend by targeting relevant users, the company deploys a machine learning model to identify key factors leading to click on ads. with AI-driven insights, the company designs targeted retention strategies:
**1. Optimize Ad Spend & Targeting**
**Action:**
Focus advertising budgets on users with a high probability of conversion (e.g., those who spend more time on-site and have moderate internet usage).
**Implementation:**
- Use custom audience targeting in platforms like Google Ads & Facebook Ads.
- Increase retargeting efforts for high-scoring users based on model predictions.

**2. Improve User Engagement for Low-Intent Visitors**
**Action:** 
Users who spend less time on-site but have high internet usage may need better engagement strategies to convert.
**Implementation:**
- Use exit-intent popups offering discounts for users with low engagement.
- Optimize website content for better navigation and product discovery.

**3. Personalize Marketing for Different Age Groups**
**Action:** 
- Young Users (18-35) → Offer fast checkout, social proof, and influencer-driven campaigns.
- Older Users (50+) → Provide detailed product descriptions, trust-building elements, and customer support. 
**Implementation:**
- A/B test different landing pages for different age segments.
- Use dynamic pricing & promotions based on engagement patterns.




## 📝 Business Impact Simulation  
A digital marketing agency, **AdTech Solutions**, partners with an e-commerce company to improve its online ad targeting strategy. The company aims to optimize its advertising budget by identifying users who are most likely to click on ads based on their demographic and behavioral characteristics.

<div align="center">
  <img src="[https://github.com/user-attachments/assets/78c8ecef-dbf8-444e-9c8c-4906da6ebf09](https://github.com/user-attachments/assets/cc56699e-e10e-4d81-9f76-fb40ddd2f7f6)" width="600">
</div>


## Acknowledgements🌟

I would like to express our deepest appreciation to Rakamin Academy for providing the opportunity to work on this exciting project. The experience and knowledge we gained throughout this journey have been invaluable.

Finally, I would like to thank those who provided their support and encouragement throughout my journey.

Regards, Bintang Phylosophie

<br>
<p align="center">
<img src="https://media1.giphy.com/media/3ohs7JG6cq7EWesFcQ/giphy.gif?cid=ecf05e47v1a2kre6ziee4vjvkc67vxhrwh8ho4089wc0aqli&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="800">
</p>


