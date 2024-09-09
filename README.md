# Predicting-Semantic-Similarity-in-Question-Pairs-Using-Machine-Learning
Developed a model to predict semantic similarity between question pairs, using XGBoost, Random Forest, vectorization, and tokenization. Addressed challenges like subjective labeling and noisy data. Exploratory analysis helped spot inconsistencies, improving accuracy and deepening my understanding of noisy datasets and human-generated labels.

Initial EDA
Here's a brief description of the dataset and analysis:

The dataset has 404,290 rows and 6 columns: `id`, `qid1`, `qid2`, `question1`, `question2`, and `is_duplicate`. There are minor missing values in the `question1` and `question2` columns. The dataset is well-balanced with 63% non-duplicate and 37% duplicate question pairs.

**Key Insights:**
- **Duplicate vs. Non-Duplicate Distribution:** The dataset contains 255,027 non-duplicate and 149,263 duplicate question pairs.
- **Unique Questions:** There are 537,933 unique questions, with 111,780 questions repeated.
- **Repetition Histogram:** A histogram shows the distribution of question repetitions, with a logarithmic y-axis to highlight frequency distribution.

This initial exploration sets the stage for further analysis and modeling by revealing data distribution and potential issues with question repetition.


I have tried different methods in this project 

Method 1
Here's a description of code and results:

**Overview:**
I have  performed a machine learning analysis to predict whether a pair of questions are duplicates using a sample of 30,000 rows from your dataset.

1. **Data Preparation:**
   - **Data Sample:** Selected 30,000 random samples from the original dataset.
   - **Missing Values:** No missing values in the sampled data.
   - **Vectorization:** Applied `CountVectorizer` to convert questions into a feature matrix with 3,000 features for each question, resulting in a combined feature matrix of 6,000 features.

2. **Modeling:**
   - **Random Forest Classifier:** 
     - **Accuracy:** 74.2% on the test set.
   - **XGBoost Classifier:** 
     - **Accuracy:** 73.3% on the test set.
     - **Warning:** The use of label encoding is deprecated, and the default evaluation metric has changed in recent XGBoost versions.

**Key Insights:**
- Both Random Forest and XGBoost models achieved comparable accuracy, indicating that either method can be effective for this classification task.
- The accuracy scores suggest that the models perform reasonably well, but there's room for improvement, possibly through hyperparameter tuning or using advanced text processing techniques.

This initial analysis provides a solid foundation for further experimentation with model optimization and additional feature engineering.


Method 2


