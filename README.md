
# Income Prediction

## Project Overview

This project aims to predict whether an individual's annual income exceeds 50K based on various census attributes. The analysis involves data cleaning, exploratory data analysis (EDA) to understand feature relationships with income, feature engineering, model training using XGBoost, hyperparameter tuning, and evaluation.

## Dataset

The dataset used is `au_test.csv` in this project. It contains demographic and employment-related information for individuals.

**Key Features (prior to one-hot encoding):**
*   `age`: Age of the individual.
*   `workclass`: Employment type (e.g., Private, Self-emp-not-inc).
*   `fnlwgt`: Final weight, a demographic weighting factor.
*   `education`: Highest education level.
*   `education-num`: Numerical representation of education level.
*   `marital-status`: Marital status.
*   `occupation`: Occupation type.
*   `relationship`: Relationship status (e.g., Husband, Own-child).
*   `race`: Race of the individual.
*   `sex`: Gender of the individual.
*   `capital-gain`: Capital gains.
*   `capital-loss`: Capital losses.
*   `hours-per-week`: Hours worked per week.
*   `native-country`: Country of origin.
*   `class`: Target variable (income <=50K or >50K).

## Workflow

1.  **Data Loading and Initial Exploration:**
    *   Loaded the dataset using pandas.
    *   Performed initial checks (`.head()`, `.tail()`, `.shape`, `.info()`).

2.  **Data Cleaning:**
    *   Identified '?' as a placeholder for missing values.
    *   Replaced '?' with `np.nan` and dropped rows with missing values.
    *   Encoded the target variable `class` into binary (0 for `<=50K.` and 1 for `>50K.`).

3.  **Feature Engineering & Preprocessing:**
    *   Separated features (X) and target (y).
    *   Applied one-hot encoding to categorical features.
    *   Split data into training (80%) and testing (20%) sets.
    *   Standardized numerical features.

4.  **Handling Class Imbalance:**
    *   Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance class distribution.

5.  **Model Training and Hyperparameter Tuning:**
    *   Trained an initial XGBoost classifier.
    *   Used `RandomizedSearchCV` for hyperparameter tuning of the XGBoost model, optimizing for `roc_auc`.
    *   Trained a final model using the best hyperparameters.

6.  **Model Evaluation:**
    *   Evaluated models using Accuracy, Classification Report, Confusion Matrix, and ROC AUC Score.

7.  **Exploratory Data Analysis (EDA) & Segment Visualization:**
    *   Visualized income distribution and its relationship with key demographic and employment features.

## Key Findings & Insights

### Model Performance:
*   The XGBoost models (both initial and tuned) demonstrated good predictive capability, with ROC AUC scores around 0.90-0.91, indicating a strong ability to distinguish between income classes.
*   Accuracy was consistently above 83%.
*   The models performed better on the majority class (<=50K) but achieved reasonable recall for the minority class (>50K), especially after SMOTE.

### Exploratory Data Analysis Insights:

*   **Income by Education Level:**
    *   There's a strong positive correlation: **higher education levels significantly increase the likelihood of earning >$50K.**
    *   Individuals with Masters, Professional School, or Doctorate degrees have the highest proportion of high earners.
    *   Those with less than a high school diploma predominantly earn <=$50K.

*   **Income by Age Group:**
    *   Income tends to increase with age, peaking in middle age.
    *   **The 36-45 and 46-55 age groups have the highest percentage of individuals earning >$50K.**
    *   Younger individuals (18-25) have a very low proportion of high earners, and the proportion also declines for those aged 65+.

*   **Income by Workclass:**
    *   **Individuals who are `Self-emp-inc` (self-employed, incorporated) have the highest percentage earning >$50K.**
    *   `Federal-gov` employees also show a relatively high proportion of high earners.
    *   Most other workclasses, including `Private` sector, have a majority earning <=$50K.

*   **Income by Occupation:**
    *   **`Exec-managerial`, `Prof-specialty`, and `Armed-Forces` occupations show the highest percentages of individuals earning >$50K.**
    *   Occupations like `Other-service`, `Handlers-cleaners`, and `Priv-house-serv` have very low proportions of high earners.

*   **Income by Gender:**
    *   A significant disparity exists. **Males have a substantially higher percentage of individuals earning >$50K compared to Females.**
    *   The vast majority of females in the dataset earn <=$50K.

## Libraries Used

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn (for `train_test_split`, `StandardScaler`, `RandomizedSearchCV`, metrics)
*   imblearn (for `SMOTE`)
*   xgboost

## How to Run

1.  Ensure all the libraries listed above are installed.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
    ```
2.  Place the `au_test.csv` file in the same directory as the Jupyter notebook or Python script.
3.  Run the Jupyter notebook cells sequentially.

## Potential Future Work

*   Explore different imputation techniques for missing values.
*   Experiment with other classification models.
*   Conduct more advanced feature engineering.
*   Investigate strategies to further improve precision or recall for specific classes based on project goals.
*   Analyze feature importances from the model in more detail.