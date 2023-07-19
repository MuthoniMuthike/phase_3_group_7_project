# SyriaTel Customer Churn Prediction

## 1.0 Project Overview

SyriaTel, a leading telecommunications company, faces the challenge of customer churn. To minimize financial losses and secure the company's future, this project aims to develop a classification model that predicts customer churn by uncovering patterns and factors associated with it. The company can therefore take proactive measures based on these insights to retain customers and ensure sustained growth.

## 2.0 Business Problem

SyriaTel Telecommunications is a company dedicated to connecting people through seamless communication. In their relentless pursuit of excellence, they have encountered a challenge of customer churn. Each lost connection not only signifies a departure but also poses a threat to the company's financial growth and future. By understanding factors and patterns associated with it and developing a classification model that predicts customer churn effectively, SyriaTel can take targeted actions to prevent churn and ensure business continuity.

## 3.0 Project Objectives

•	To build a robust predictive model using the provided dataset to classify customers as churned or not churned.
•	To identify any discernible patterns or trends associated with customer churn.
•	To provide actionable insights to SyriaTel to take proactive measures to retain customers once the model is developed.
•	To evaluate the model performance using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.

## 4.0 Data Preparation

### 4.1 Feature Engineering

We created two new features: 'call_rates_day' and 'call_rates_night' to represent the ratio of the total day charges to the total day minutes and the ratio of the total night charges to the total night minutes respectively. This would help us determine if the charges incurred during day and night calls might be contributing factors to customer churn.

## 5.0 Exploratory Data Analysis

### 5.1 Univariate Analysis

#### 5.1.1. Numerical variables
•	For the numerical variables, univariate analysis, we plotted histograms to provide insights into the frequency distribution of our features We noticed that most features in our dataset such as 'account length', 'total day minutes', 'total day calls', 'total day charge' and others had a somewhat normal distribution while 'customer service calls' and 'total intl calls' were positively skewed.
•	We also plotted box plots to check for outliers in our features in which we found most of our features to exhibit a range of extreme values. These values represented the range of variability in the data for each corresponding feature. While this range of variability could be considered as outliers, we deemed them admissible since they accurately reflected real-world scenarios in the telecommunications context.
5.1.2. Categorical variables
•	For the categorical variables, univariate analysis, we plotted bar plots to get a sense of the distribution of our categorical data. The bar plots suggest only a minority of the customers are on the international plan and the voice mail plan with only a few customers having these plans. It will be interesting to gain insights on how this is a contributing factor to customer churn.
•	For our target feature 'churn', our visualization indicates the presence of class imbalance because it shows a significant difference between the false and true class with false being the majority.

### 5.2 Bivariate Analysis

#### 5.2.1. Numerical variables
•	For our numerical variables, Bivariate analysis, we plotted scatter plots to visualize the relationship between the numerical variables against customer churn. According to our plots, the relationship is non-linear making a classification model most appropriate for this dataset to help make predictions and discover underlying patterns and relationships in the data.

#### 5.2.2. Categorical variables
•	For our categorical variables, Bivariate analysis, the bar plots represent the relationship between the categorical variables "international plan" and "voicemail plan" against customer churn, helping us understand how these variables are related to the likelihood of customer churn.
•	Our visualization indicates customers with an international plan have a higher churn rate to those without, while customers with no voice mail plan tend to have a higher churn rate to those with a plan.

### 5.3 Multivariate Analysis
•	Generated a heatmap to show the correlation coefficients between pairs of numerical variables in our dataset. The red labels represent high values in correlation between our variables which is an indication of the presence of multicollinearity.

## 6.0 Modelling
To begin modelling we performed the following steps:
•	Assigned the 'churn' column to y
•	Dropped the 'churn', 'phone number' and 'state' columns from df, and assign the resulting Data Frame to X
•	Performed one-hot encoding on the categorical features.
•	Split X and y into training and test sets, assigned 20% to the test set and set the random_state to 42
•	Recall will be our main performance evaluation metric because our project objective is to help SyriaTel Telecommunications identify as many churned customers as possible to help gain insights into the reasons for churn. Recall evaluation metric will ensure that a higher proportion of actual churned customers are correctly identified, allowing for a more comprehensive analysis of their characteristics and behaviors.

### 6.1 Model 1: KNN Classifier
#### 6.1.1. Baseline Model
For the project, the KNN Classifier was selected as the first model.
The results from the baseline KNN Classifier model were as follows:
•	A recall of 0.3176 means that approximately 31.8% of the actual churned customers are correctly identified by the baseline model as churned. This is fairly better than the cross-validation recall score of 23.65% meaning the model has the ability to generalize unseen data.
•	An accuracy of 0.9040 suggests 90.4% of the predictions made by the model on the test set are correct while a precision of 0.8182 suggests 81.8% of the customers predicted as churned by the baseline model are actually churned customers.
•	The F1-score being a harmonic mean of precision and recall, considers both false positives and false negatives. In this case, the F1-score of 0.4576 indicates the baseline model's performance is moderate, taking into account both precision and recall.

#### 6.1.2. KNN Model Improvement
The following areas were addressed to improve the baseline model:
•	Dealing with class imbalance using SMOTE
•	Hyperparameter Tuning

#### 6.1.3. Improved KNN Model Results
•	After the using SMOTE to balance the data and tuning of hyperparameters using GridSearchCV, Recall being our main performance metric greatly improved to 0.7647 from the initial 0.31764 score of the baseline model. 
•	This means out of all actual churned customers; this model can accurately predict 76.7% of them as churned. While this is an okay percentage, we will look into other models to see if we can attain a higher recall for SyriaTel to work with.

### 6.2 Model 2: Logistic Regression

#### 6.2.1. Baseline Model
For the Logistic Regression baseline model:
•	A recall of 0.2235 means that approximately 22.35% of the actual churned customers are correctly identified by the baseline model as churned. Against, the cross-validation recall score of 21.14%, the model indicates an okay ability to generalize to new, unseen data.
•	An accuracy of 0.8801 suggests 88.01% of the predictions made by the model on the test set are correct and a precision of 0.5758 suggests 57.58% of the customers predicted as churned by the baseline model are actually churned customers.
•	The F1-score being a harmonic mean of precision and recall, considers both false positives and false negatives. In this case, the F1-score of 0.322 indicates the baseline model's performance is below average, taking into account both precision and recall.
•	Improvements can be made by handling class imbalance and performing regularization. This will help optimize the model and ensure churn patterns are captured more accurately.

#### 6.2.2. Logistic Regression Model Improvement
The following areas were addressed to improve the baseline model:
•	Applying class weights to help handle class imbalance. This will help give higher weightage to the minority class during model training.
•	Hyperparameter Tuning

#### 6.2.3. Improved Logistic Regression Model Results
By addressing class imbalance and accounting for regularization, the recall performance metric has greatly improved from 0.2235 to 0.7647. This means out of all actual churned customers; this model can predict up to 76.5% as churned.


### 6.3 Model 3: Decision Tree Classifier

#### 6.3.1. Baseline Model
For the Decision Tree classifier:
•	A recall 0.6824 suggests that of all actually churned customers, the model correctly identifies approximately 68.24% of them as churned. This is the best recall metric of all our baseline models so far and we will therefore strive to improve the model further to capture churn patterns even more accurately.
•	The cross-validation recall score of 0.7256 means that on average, the model achieved a recall of 72.56% across the 5 cross-validation folds. This indicates the model generalizes pretty well to unseen data.
•	An accuracy of 0.9175 means that the model predicts the correct churn status for approximately 91.75% of the instances in the test set. A precision of 0.6744 also indicates that out of all the instances predicted as churn, approximately 67.44% are churned customers.
•	The F1-score being the harmonic mean of precision and recall, a score of 0.6784, indicates a good balance between precision and recall.

#### 6.3.2. Decision Tree Classifier Model Improvement
The following areas were addressed to improve the baseline model:
•	Dealing with class imbalance using SMOTE
•	Hyperparameter Tuning. We took the approach of tuning one hyperparameter at a time to optimize our model performance. We will be seeking to optimize four hyperparameters; min_samples_splits, min_samples_leaf, max_depth and max_features

#### 6.3.3. Improved Decision Tree Classifier Model Results
•	After improving the baseline decision tree classifier by tuning the min_samples_splits, min_samples_leaf, max_depth and max_features hyperparameters and using SMOTE to address class imbalance, our recall had a significant increase from 68.24% in the vanilla classifier to 80% in the optimized model. With an F1-score of 0.7311, the enhanced model strikes a balance between precision and recall.
•	This trade-off could be acceptable in this scenario, as the main goal is to capture as many churned customers as possible for SyriaTel Telecommunications.
•	With the enhanced model, SyriaTel Telecommunications will have a 80% chance of identifying more customers who are likely to churn. This can provide valuable insights into the reasons for churn, allowing the company to take proactive measures to retain these customers and improve their overall customer satisfaction.

### 6.4 Model 4: XG Boost 

#### 6.4.1. Baseline Model 
For the XG Boost baseline model:
•	A recall score is 0.8118, which means the model is very effective at capturing a significant portion of actual churned customers.
•	An accuracy of 96.1% suggests that the model is making correct predictions most of the time while a precision score of 0.8734, indicates that the model is very good at identifying customers who are likely to churn. This is essential for the project objective of identifying as many churned customers as possible.
•	An F1-score is 0.8415 shows a good balance between precision and recall.
•	Overall, the XG Boost Classifier outperforms the Enhanced Decision Tree Classifier in terms of accuracy, precision, recall, and F1-score. It is a highly effective model for identifying churned customers and can provide valuable insights into customer behavior and churn patterns. This model is well-suited for the project objective of maximizing the identification of churned customers, allowing for a comprehensive analysis of the reasons for churn and the characteristics of churned customers.

#### 6.4.2. XG Boost Model Improvement
The model was tuned to increase the model performance and avoid overfitting

#### 6.4.3. Improved XG Boost Model Results
For the Tuned XG Boost Classifier:
•	The recall score for the Tuned XG Boost Classifier at 0.8235 is the best one yet and aligns with our project objective, which is to identify as many churned customers as possible to gain insights into the reasons for churn. A high recall score of 0.8235 ensures that an 82.35& of actual churned customers are correctly identified by the model.
•	A high accuracy of, indicates that the model is making correct predictions with a high rate of success. while a precision score of 0.9091, which is very high and suggests that the model is effective in identifying customers who are likely to churn, with a high proportion of true positives.
•	An F1-score of 0.8642, shows a great balance between precision and recall.

## 7.0 Model Recommendation
•	Utilize the improved XG Boost classifier as the primary model for customer churn prediction. It demonstrated the highest performance in accurately identifying churned customers.

## 8.0 Conclusion
•	Data Analysis: The analysis of SyriaTel's customer churn dataset revealed important insights about the factors influencing churn. Several visualizations, such as box plots, bar plots, scatter plots, and correlation heatmaps, provided a comprehensive understanding of the relationships between variables and their impact on churn.
•	KNN Model: The KNN classifier, although easy to implement, showed limited performance in predicting customer churn. It achieved moderate accuracy, precision, recall, and F1-score values. Therefore, it may not be the most suitable model for accurate churn prediction in this scenario.

•	Logistic Regression Model: The logistic regression model, both in its baseline and improved forms, showed better performance than the KNN model. It achieved higher accuracy, precision, recall, and F1-score values. The improved logistic regression model with tuned hyperparameters demonstrated superior performance and can be considered a good option for churn prediction.
•	Decision Tree Model: The baseline decision tree classifier achieved high accuracy, precision, recall, and F1-score values, indicating its potential for churn prediction. By tuning the hyperparameters, the decision tree model's performance improved further, achieving even higher accuracy, precision, recall, and F1-score values. The improved decision tree classifier can be considered a reliable model for customer churn prediction.
•	XG Boost Model: The XG Boost classifier, both in its baseline and improved forms, outperformed all other models in terms of accuracy, precision, recall, and F1-score. The improved XG Boost model achieved the highest performance, indicating its effectiveness in identifying churned customers accurately. It can be considered the top-performing model for customer churn prediction in this scenario.

## 9.0 Recommendation
•	Utilize the improved XG Boost classifier as the primary model for customer churn prediction. It demonstrated the highest performance in accurately identifying churned customers.
•	SyriaTel can focus on promoting and enhancing voicemail plan offerings to reduce churn.
•	Improving customer service quality and addressing issues promptly can help reduce churn.
•	Understanding the correlation between call usage and churn can help SyriaTel develop personalized plans and offers to keep customers engaged.
## 10.0 Next Step
•	Develop customer retention strategies as recommended
•	Segment customers according to their unique characteristics.
•	Improve service quality to minimize rate of churn
•	Establish a customer Feedback loop for sentiment analysis
•	Regularly assess the effectiveness of implemented strategies and initiatives by tracking key metrics and comparing them against established benchmarks.






```python

```
