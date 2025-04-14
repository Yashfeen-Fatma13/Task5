# Task5

Summary:
Exploratory Data Analysis (EDA) Report: Titanic Dataset

Objective

The primary goal of this analysis is to explore the Titanic dataset using statistical and visual methods to identify meaningful patterns, relationships, and insights that can inform further analysis or model development.


---

1. Data Loading and Overview

We began by loading the Titanic dataset from Seaborn’s built-in dataset, as it contains the necessary survived column which was missing in other sources. The dataset contains demographic and voyage information about passengers aboard the Titanic.

import seaborn as sns
df = sns.load_dataset('titanic')


---

2. Basic Exploration and Structure

To understand the structure and contents of the dataset, we used:

df.info() to check column types and null values

df.describe() to get statistical summaries of numeric columns

df.isnull().sum() to identify missing data

df['survived'].value_counts() to understand the distribution of the target variable


Observations:

The dataset contains 891 entries with 15 features.

Missing values were present in columns like age, embarked, and deck.

The survived column is imbalanced (more non-survivors than survivors).



---

3. Univariate Analysis

We visualized individual feature distributions to understand their characteristics.

Survival Count

sns.countplot(x='survived', data=df)

More passengers did not survive than those who did.


Gender Distribution

sns.countplot(x='sex', hue='survived', data=df)

Females had a significantly higher survival rate compared to males.


Passenger Class

sns.countplot(x='pclass', hue='survived', data=df)

Passengers from 1st class had the highest survival rate, while 3rd class had the lowest.


Age Distribution

sns.histplot(df['age'], bins=30, kde=True)

Majority of passengers were between 20 and 40 years old.



---

4. Bivariate Analysis

Survival vs. Age

sns.boxplot(x='survived', y='age', data=df)

Younger passengers showed a slightly higher survival tendency.


Correlation Heatmap

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')

Survival is weakly correlated with fare and class.


Pairwise Relationships

sns.pairplot(df[['survived', 'age', 'fare', 'pclass']], hue='survived')

Visualizes how numeric features interact in relation to survival

5. Missing Data Handling

Though not fully imputed, we identified key columns with missing values (like age, deck, embarked) and suggested strategies like filling with median/mode or dropping if necessary.


---

6. Summary of Insights

Gender and class were strong indicators of survival.

Younger and wealthier passengers had higher survival rates.

Significant number of missing values in age, deck, and embark_town.

The dataset is suitable for building classification models with appropriate preprocessing.
