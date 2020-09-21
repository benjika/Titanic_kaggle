import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("train.csv")
# train_data = sns.load_dataset('titanic')
test_data = pd.read_csv("test.csv")

# get info about the data
# print(train_data.info())
# print(train_data.info())
# print(train_data.describe())
# print(test_data.describe())
# print(train_data['Name'])
# print(test_data.head())

# check if there are any null values
# print(train_data.isnull().sum())
# print(test_data.isnull().sum())

# Too many null values in Cabin column
# train_data = train_data.drop(['Cabin', 'Ticket'], axis=1)
train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)

train_data['Title'] = train_data.apply(lambda x: x['Name'].split(', ', 1)[1].split(' ', 1)[0], axis=1)
test_data['Title'] = test_data.apply(lambda x: x['Name'].split(', ', 1)[1].split(' ', 1)[0], axis=1)

Title_Dictionary = {
    "Capt.": "Officer",
    "Col.": "Officer",
    "Major.": "Sir",
    "Jonkheer.": "Royalty",
    "Don.": "Royalty",
    "Dona.": "Royalty",
    "Sir.": "Sir",
    "Dr.": "Officer",
    "Rev.": "Officer",
    "the": "Royalty",
    "Mme.": "Mrs",
    "Mlle.": "Miss",
    "Ms.": "Mrs",
    "Mr.": "Mr",
    "Mrs.": "Mrs",
    "Miss.": "Miss",
    "Master.": "Master",
    "Lady.": "Royalty"
}

train_data['Title'] = train_data['Title'].map(Title_Dictionary)
test_data['Title'] = test_data['Title'].map(Title_Dictionary)

train_data['Age'] = train_data.apply(
    (lambda x: train_data[train_data['Title'] == x['Title']]['Age'].mean(skipna=True) if pd.isnull(x['Age']) else x[
        'Age']), axis=1)

test_data['Age'] = test_data.apply(
    (lambda x: test_data[test_data['Title'] == x['Title']]['Age'].mean(skipna=True) if pd.isnull(x['Age']) else x[
        'Age']), axis=1)

train_data['Embarked'] = train_data['Embarked'].fillna(test_data['Embarked'].value_counts().index[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].value_counts().index[0])
train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0})
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})
train_data['Embarked'] = train_data['Embarked'].map({'Q': 2, 'C': 1, 'S': 0})
test_data['Embarked'] = test_data['Embarked'].map({'Q': 2, 'C': 1, 'S': 0})

oe = LabelEncoder().fit(pd.concat([train_data['Title'], test_data['Title']]).unique())
# train_data['Title'] = temp_titles.index(train_data['Title']

print(len(train_data[(train_data['Age'] > 50) & (train_data['Survived'] == 1)]) / len(
    train_data[(train_data['Age'] > 50)]))

train_data['Title'] = oe.transform(train_data['Title'].to_list())
test_data['Title'] = oe.transform(test_data['Title'].to_list())

train_data = train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
X = train_data.loc[:, 'Pclass':]
y = train_data['Survived']
rand_for_clf = RandomForestClassifier(n_estimators=100, max_features=4, max_depth=3)
rand_for_clf = rand_for_clf.fit(X, y)
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = rand_for_clf.feature_importances_
print(features.sort_values(['importance']))
sol = pd.read_csv('gender_submission.csv')
sol['Survived'] = rand_for_clf.predict(test_data)
sol.to_csv('my_submission.csv', index=False)
