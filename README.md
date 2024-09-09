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

This code outlines the process of building a machine learning model to classify duplicate and non-duplicate questions from a dataset. Here's a breakdown of the steps involved:

### 1. **Data Exploration & Sampling**:
   - The dataset contains pairs of questions, and the goal is to predict whether they are duplicates.
   - A sample of 30,000 questions is taken from the full dataset (404,290 samples).
   - There are no missing or duplicate values in the dataset, making it ready for further analysis.

### 2. **Target Distribution**:
   - The target variable `is_duplicate` has two classes:
     - 0 (non-duplicate): 63.37%
     - 1 (duplicate): 36.62%
   - This class imbalance will be taken into account in modeling.

### 3. **Repeated Questions Analysis**:
   - The number of unique questions is 55,299, and 3,480 questions are repeated.

### 4. **Feature Engineering**:
   - New features are created based on the questions:
     - `q1_len` & `q2_len`: Length of each question.
     - `q1_num_words` & `q2_num_words`: Number of words in each question.
     - `word_common`: Number of common words between the two questions.
     - `word_total`: Total number of unique words between the two questions.
     - `word_share`: The proportion of common words between the two questions.

### 5. **Text Vectorization**:
   - Using `CountVectorizer`, the questions are transformed into a bag-of-words representation with 3,000 features for each question, resulting in a combined feature space of 6,000 columns.

### 6. **Training a Classifier**:
   - After concatenating the engineered features, the final dataset contains 6,008 features.
   - A Random Forest classifier is trained, achieving an accuracy of **76.83%**.
   - An XGBoost model is also tested, achieving an accuracy of **76.45%**.

### 7. **Advanced Features**:
   The code also mentions advanced features that can be computed for better prediction:
   - **Token-based features**: 
     - `cwc_min`, `cwc_max`: Ratio of common words to the length of the smaller/larger question.
     - `csc_min`, `csc_max`: Ratio of common stop words to the smaller/larger stop word count.
     - `ctc_min`, `ctc_max`: Ratio of common tokens to the smaller/larger token count.
     - `last_word_eq`, `first_word_eq`: Check if the first/last word is the same.
   - **Length-based features**:
     - `mean_len`, `abs_len_diff`, `longest_substr_ratio`: Statistical measures of word lengths and substrings.
   - **Fuzzy features** (from `fuzzywuzzy` library):
     - `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, `token_set_ratio`: Measures of similarity between question pairs.

These features aim to capture the structure, content, and similarity of the two questions for better classification performance.

Method 3

Text Preprocessing

The code starts by loading a dataset of question pairs and their corresponding labels (duplicate or not). It then applies various preprocessing techniques to the text data, including:

Tokenization: splitting the text into individual words or tokens.
Stopword removal: removing common words like "the", "and", etc. that do not add much value to the meaning of the text.
Stemming or Lemmatization: reducing words to their base form (e.g., "running" becomes "run").
Vectorization: converting the text data into numerical vectors that can be processed by machine learning algorithms.
Feature Engineering

The code extracts various features from the preprocessed text data, including:

Basic features: length of the questions, number of words, common words, etc.
Token-based features: features extracted from the tokens, such as token frequency, token co-occurrence, etc.
Fuzzy matching features: features that measure the similarity between the two questions, such as Levenshtein distance, Jaro-Winkler distance, etc.
Bag-of-words (BoW) features: features that represent the frequency of each word in the questions.
Model Training

The code trains a Random Forest Classifier on the extracted features to predict whether a pair of questions is duplicate or not.

Model Deployment

The trained model is saved to disk using the pickle module, which allows it to be loaded and used in the future without having to retrain it.

Query Point Creator

The code defines a function query_point_creator that takes a pair of questions as input, preprocesses them, and creates a feature vector that can be used to make predictions with the trained model.

Example Usage

The code provides an example of using the query_point_creator function to create a feature vector for a pair of questions and then using the trained model to predict whether they are duplicate or not.

Overall, the code provides a comprehensive implementation of a text classification system for duplicate question detection, including text preprocessing, feature engineering, model training, and model deployment.






