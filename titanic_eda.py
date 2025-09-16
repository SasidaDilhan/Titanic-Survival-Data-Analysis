import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/train.csv")

# Check data
print(df.head())
print(df.info())
print(df.isnull().sum())

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Visualization
sns.countplot(x="Survived", data=df)
plt.show()

sns.barplot(x="Sex", y="Survived", data=df)
plt.show()

sns.barplot(x="Pclass", y="Survived", data=df)
plt.show()
