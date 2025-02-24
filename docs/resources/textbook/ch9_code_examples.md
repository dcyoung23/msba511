## Chapter 9 Classification and Regression Trees

**Original Code Credit:**: Shmueli, Galit; Bruce, Peter C.; Gedeck, Peter; Patel, Nitin R.. Data Mining for Business Analytics Wiley.

*Modifications* have been made from the original textbook examples due to version changes in library dependencies and/or for clarity.

Download this notebook and data [**here**](https://github.com/dcyoung23/msba511/tree/main/resources/examples).

### Import Libraries


```python
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary
import matplotlib

%matplotlib inline
```

    no display found. Using non-interactive Agg backend
    

### 9.2 Classification Trees

#### Example: Riding Mowers


```python
mower_df = pd.read_csv(os.path.join('data', 'RidingMowers.csv'))
# use max_depth to control tree size (None = full tree)
classTree = DecisionTreeClassifier(random_state=0, max_depth=1)
classTree.fit(mower_df.drop(columns=['Ownership']), mower_df['Ownership'])
print("Classes: {}".format(', '.join(classTree.classes_)))
plotDecisionTree(classTree, feature_names=mower_df.columns[:2], 
                 class_names=classTree.classes_)
```

    Classes: Nonowner, Owner
    




    
![png](assets\ch9_output_6_1.png)
    



### 9.3 Evaluating the Performance of a Classification Tree

#### Example 2: Acceptance of Personal Loan



```python
bank_df = pd.read_csv(os.path.join('data', 'UniversalBank.csv'))
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)

X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

fullClassTree = DecisionTreeClassifier(random_state=1)
fullClassTree.fit(train_X, train_y)

plotDecisionTree(fullClassTree, feature_names=train_X.columns)

```




    
![png](assets\ch9_output_9_0.png)
    




```python
classificationSummary(train_y, fullClassTree.predict(train_X))
classificationSummary(valid_y, fullClassTree.predict(valid_X))
```

    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual    0    1
         0 2713    0
         1    0  287
    Confusion Matrix (Accuracy 0.9790)
    
           Prediction
    Actual    0    1
         0 1790   17
         1   25  168
    


```python
treeClassifier = DecisionTreeClassifier(random_state=1)

scores = cross_val_score(treeClassifier, train_X, train_y, cv=5)
print('Accuracy scores of each fold: ', [f'{acc:.3f}' for acc in scores])

```

    Accuracy scores of each fold:  ['0.988', '0.973', '0.993', '0.982', '0.993']
    

### 9.4 Avoiding Overfitting



```python
smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20,
                        min_impurity_decrease=0.01, random_state=1)
smallClassTree.fit(train_X, train_y)

plotDecisionTree(smallClassTree, feature_names=train_X.columns)

```




    
![png](assets\ch9_output_13_0.png)
    




```python
classificationSummary(train_y, smallClassTree.predict(train_X))
classificationSummary(valid_y, smallClassTree.predict(valid_X))

```

    Confusion Matrix (Accuracy 0.9823)
    
           Prediction
    Actual    0    1
         0 2711    2
         1   51  236
    Confusion Matrix (Accuracy 0.9770)
    
           Prediction
    Actual    0    1
         0 1804    3
         1   43  150
    


```python
# Start with an initial guess for parameters
param_grid = {
    'max_depth': [10, 20, 30, 40], 
    'min_samples_split': [20, 40, 60, 80, 100], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5, 
                          n_jobs=-1)  # n_jobs=-1 will utilize all available CPUs 
gridSearch.fit(train_X, train_y)
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)

# Adapt grid based on result from initial grid search
param_grid = {
    'max_depth': list(range(2, 16)),  # 14 values 
    'min_samples_split': list(range(10, 22)), # 11 values
    'min_impurity_decrease': [0.0009, 0.001, 0.0011], # 3 values
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5, 
                          n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)

bestClassTree = gridSearch.best_estimator_
```

    Initial score:  0.9876666666666667
    Initial parameters:  {'max_depth': 10, 'min_impurity_decrease': 0.0005, 'min_samples_split': 20}
    Improved score:  0.9873333333333333
    Improved parameters:  {'max_depth': 4, 'min_impurity_decrease': 0.0011, 'min_samples_split': 13}
    


```python
# fine-tuned tree: training
classificationSummary(train_y, bestClassTree.predict(train_X))
```

    Confusion Matrix (Accuracy 0.9867)
    
           Prediction
    Actual    0    1
         0 2708    5
         1   35  252
    


```python
classificationSummary(valid_y, bestClassTree.predict(valid_X))
```

    Confusion Matrix (Accuracy 0.9815)
    
           Prediction
    Actual    0    1
         0 1801    6
         1   31  162
    


```python
plotDecisionTree(bestClassTree, feature_names=train_X.columns)
```




    
![png](assets\ch9_output_18_0.png)
    



### 9.7 Regression Trees



```python
from sklearn.tree import DecisionTreeRegressor
toyotaCorolla_df = pd.read_csv(os.path.join('data', 'ToyotaCorolla.csv')).iloc[:1000,:]
toyotaCorolla_df = toyotaCorolla_df.rename(columns={'Age_08_04': 'Age', 'Quarterly_Tax': 'Tax'})

predictors = ['Age', 'KM', 'Fuel_Type', 'HP', 'Met_Color', 'Automatic', 'CC', 
              'Doors', 'Tax', 'Weight']
outcome = 'Price'

X = pd.get_dummies(toyotaCorolla_df[predictors], drop_first=True)
y = toyotaCorolla_df[outcome]

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

# user grid search to find optimized tree
param_grid = {
    'max_depth': [5, 10, 15, 20, 25], 
    'min_impurity_decrease': [0, 0.001, 0.005, 0.01], 
    'min_samples_split': [10, 20, 30, 40, 50], 
}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Initial parameters: ', gridSearch.best_params_)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
    'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008], 
    'min_samples_split': [14, 15, 16, 18, 20, ], 
}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved parameters: ', gridSearch.best_params_)

regTree = gridSearch.best_estimator_

regressionSummary(train_y, regTree.predict(train_X))
regressionSummary(valid_y, regTree.predict(valid_X))


```

    Initial parameters:  {'max_depth': 5, 'min_impurity_decrease': 0, 'min_samples_split': 20}
    Improved parameters:  {'max_depth': 6, 'min_impurity_decrease': 0, 'min_samples_split': 16}
    
    Regression statistics
    
                          Mean Error (ME) : 0.0000
           Root Mean Squared Error (RMSE) : 1058.8202
                Mean Absolute Error (MAE) : 767.7203
              Mean Percentage Error (MPE) : -0.8074
    Mean Absolute Percentage Error (MAPE) : 6.8325
    
    Regression statistics
    
                          Mean Error (ME) : 60.5241
           Root Mean Squared Error (RMSE) : 1554.9146
                Mean Absolute Error (MAE) : 1026.3487
              Mean Percentage Error (MPE) : -1.3082
    Mean Absolute Percentage Error (MAPE) : 9.2311
    

### 9.8 Improving Prediction: Random Forests and Boosted Trees



```python
X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_X, train_y)

# variable (feature) importance plot
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

df = pd.DataFrame({'feature': train_X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance')
print(df)

ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')
plt.show()

# confusion matrix for validation set
classificationSummary(valid_y, rf.predict(valid_X))

```

                   feature  importance       std
    7   Securities Account    0.003882  0.004752
    9               Online    0.006406  0.005354
    10          CreditCard    0.007642  0.006951
    6             Mortgage    0.034209  0.023448
    1           Experience    0.035575  0.016021
    0                  Age    0.036275  0.015869
    8           CD Account    0.057934  0.043198
    3               Family    0.111388  0.053174
    4                CCAvg    0.171907  0.102980
    5            Education    0.200715  0.100976
    2               Income    0.334067  0.129097
    


    
![png](assets\ch9_output_22_1.png)
    


    Confusion Matrix (Accuracy 0.9820)
    
           Prediction
    Actual    0    1
         0 1803    4
         1   32  161
    


```python
boost = GradientBoostingClassifier()
boost.fit(train_X, train_y)
classificationSummary(valid_y, boost.predict(valid_X))

```

    Confusion Matrix (Accuracy 0.9835)
    
           Prediction
    Actual    0    1
         0 1799    8
         1   25  168
    


```python

```
