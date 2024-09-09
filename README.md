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

