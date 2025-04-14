## Chapter 13 Combining Methods: Ensembles and Uplift Modeling

**Original Code Credit:**: Shmueli, Galit; Bruce, Peter C.; Gedeck, Peter; Patel, Nitin R.. Data Mining for Business Analytics Wiley.

*Modifications* have been made from the original textbook examples due to version changes in library dependencies and/or for clarity.

Download this notebook and data [**here**](https://github.com/dcyoung23/msba511/tree/main/resources/examples).

### Import Libraries


```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from dmba import classificationSummary

import matplotlib

%matplotlib inline

SEED = 1
```

    no display found. Using non-interactive Agg backend
    

### 13.1 Ensembles

**Example Of Bagging and Boosting Classification Trees (Personal Loan Data)**


```python
bank_df = pd.read_csv(os.path.join('data', 'UniversalBank.csv'))
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)
# split into training and validation
X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, 
                                                      random_state=3)
# single tree
defaultTree = DecisionTreeClassifier(random_state=SEED)
defaultTree.fit(X_train, y_train)
classes = defaultTree.classes_
classificationSummary(y_valid, defaultTree.predict(X_valid), class_names=classes)
# bagging 
bagging = BaggingClassifier(DecisionTreeClassifier(random_state=SEED), 
                            n_estimators=100, random_state=SEED)
bagging.fit(X_train, y_train)
classificationSummary(y_valid, bagging.predict(X_valid), class_names=classes)
# boosting
boost = AdaBoostClassifier(DecisionTreeClassifier(random_state=SEED), 
                            n_estimators=100, random_state=SEED)
boost.fit(X_train, y_train)
classificationSummary(y_valid, boost.predict(X_valid), class_names=classes)

```

    Confusion Matrix (Accuracy 0.9825)
    
           Prediction
    Actual    0    1
         0 1778   15
         1   20  187
    Confusion Matrix (Accuracy 0.9855)
    
           Prediction
    Actual    0    1
         0 1781   12
         1   17  190
    Confusion Matrix (Accuracy 0.9835)
    
           Prediction
    Actual    0    1
         0 1776   17
         1   16  191
    

### 13.2 Uplift (Persuasion) Modeling

**Uplift in Python Applied to the Voters Data**


```python
voter_df = pd.read_csv(os.path.join('data', 'Voter-Persuasion.csv'))
# Preprocess data frame
predictors = ['AGE', 'NH_WHITE', 'COMM_PT', 'H_F1', 'REG_DAYS', 
            'PR_PELIG', 'E_PELIG', 'POLITICALC', 'MESSAGE_A']
outcome = 'MOVED_AD'
classes = list(voter_df.MOVED_AD.unique())
# Partition the data
X = voter_df[predictors]
y = voter_df[outcome]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.40, 
                                                      random_state=1)
# Train a random forest classifier using the training set
rfModel = RandomForestClassifier(n_estimators=100, random_state=1)
rfModel.fit(X_train, y_train)
# Calculating the uplift
uplift_df = X_valid.copy()  # Need to create a copy to allow modifying data
uplift_df.MESSAGE_A = 1
predTreatment = rfModel.predict_proba(uplift_df)
uplift_df.MESSAGE_A = 0
predControl = rfModel.predict_proba(uplift_df)
upliftResult_df = pd.DataFrame({
    'probMessage': predTreatment[:,1],
    'probNoMessage': predControl[:,1],
    'uplift': predTreatment[:,1] - predControl[:,1],
    }, index=uplift_df.index)
upliftResult_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>probMessage</th>
      <th>probNoMessage</th>
      <th>uplift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9953</th>
      <td>0.78</td>
      <td>0.64</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>3850</th>
      <td>0.37</td>
      <td>0.39</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>4962</th>
      <td>0.21</td>
      <td>0.13</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>3886</th>
      <td>0.84</td>
      <td>0.58</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>5437</th>
      <td>0.14</td>
      <td>0.25</td>
      <td>-0.11</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
