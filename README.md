# Titanic Survival Prediction using Naive Bayes

This project uses the Naive Bayes algorithm to predict survival from the Titanic crash based on passenger data.

## Dataset
The dataset used is the Titanic dataset, which includes the following features:
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `Fare`: Passenger fare
- `Survived`: Survival (0 = No, 1 = Yes)

## Steps Involved

### 1. Data Preprocessing
```python
import pandas as pd

# Load the dataset
df = pd.read_csv("titanic.csv")

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

# Convert categorical 'Sex' column into dummy variables
dummies = pd.get_dummies(df.Sex)
inputs = pd.concat([df.drop('Sex', axis='columns'), dummies], axis='columns')
inputs.drop(['male'], axis='columns', inplace=True)  # Avoid dummy variable trap

# Handle missing values in 'Age'
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

# Features and Target
inputs = inputs.drop('Survived', axis='columns')
target = df.Survived

#Model Training....
