## Chapter 14 Association Rules and Collaborative Filtering

**Original Code Credit:**: Shmueli, Galit; Bruce, Peter C.; Gedeck, Peter; Patel, Nitin R.. Data Mining for Business Analytics Wiley.

*Modifications* have been made from the original textbook examples due to version changes in library dependencies and/or for clarity.

Download this notebook and data [**here**](https://github.com/dcyoung23/msba511/tree/main/resources/examples).

### Import Libraries


```python
import os
import heapq
import random
from collections import defaultdict
import pandas as pd

import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
```

### 14.1 Association Rules

#### Example 1: Synthetic Data on Purchases of Phone Faceplates


```python
# Load and preprocess data set 
fp_df = pd.read_csv(os.path.join('data', 'Faceplate.csv'))
fp_df.set_index('Transaction', inplace=True)
fp_df = fp_df.astype(bool, 0)

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2, use_colnames=True)

# convert into rules
rules = association_rules(itemsets, num_itemsets=len(fp_df), metric='confidence', min_threshold=0.5)
rules.sort_values(by=['lift'], ascending=False).head(6)
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>representativity</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
      <th>jaccard</th>
      <th>certainty</th>
      <th>kulczynski</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>(Red, White)</td>
      <td>(Green)</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.5</td>
      <td>2.500000</td>
      <td>1.0</td>
      <td>0.12</td>
      <td>1.6</td>
      <td>1.000</td>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(Green)</td>
      <td>(Red, White)</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>2.500000</td>
      <td>1.0</td>
      <td>0.12</td>
      <td>inf</td>
      <td>0.750</td>
      <td>0.500000</td>
      <td>1.000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Green)</td>
      <td>(Red)</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.666667</td>
      <td>1.0</td>
      <td>0.08</td>
      <td>inf</td>
      <td>0.500</td>
      <td>0.333333</td>
      <td>1.000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(White, Green)</td>
      <td>(Red)</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.666667</td>
      <td>1.0</td>
      <td>0.08</td>
      <td>inf</td>
      <td>0.500</td>
      <td>0.333333</td>
      <td>1.000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Orange)</td>
      <td>(White)</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.428571</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>inf</td>
      <td>0.375</td>
      <td>0.285714</td>
      <td>1.000</td>
      <td>0.642857</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(Green)</td>
      <td>(White)</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.428571</td>
      <td>1.0</td>
      <td>0.06</td>
      <td>inf</td>
      <td>0.375</td>
      <td>0.285714</td>
      <td>1.000</td>
      <td>0.642857</td>
    </tr>
  </tbody>
</table>
</div>



#### Example 2: Rules for Similar Book Purchases


```python
# load dataset
all_books_df = pd.read_csv(os.path.join('data', 'CharlesBookClub.csv'))
ignore = ['Seq#', 'ID#', 'Gender', 'M', 'R', 'F', 'FirstPurch', 'Related Purchase',
          'Mcode', 'Rcode', 'Fcode', 'Yes_Florence', 'No_Florence']
count_books = all_books_df.drop(columns=ignore)
count_books[count_books > 0] = 1
count_books = count_books.astype(bool, 0)
# create frequent itemsets and rules
itemsets = apriori(count_books, min_support=200/4000, use_colnames=True)
rules = association_rules(itemsets, num_itemsets=len(count_books), metric='confidence', min_threshold=0.5)
rules.sort_values(by=['lift'], ascending=False).head(10)
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>representativity</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
      <th>jaccard</th>
      <th>certainty</th>
      <th>kulczynski</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>(YouthBks, RefBks)</td>
      <td>(ChildBks, CookBks)</td>
      <td>0.08125</td>
      <td>0.24200</td>
      <td>0.05525</td>
      <td>0.680000</td>
      <td>2.809917</td>
      <td>1.0</td>
      <td>0.035588</td>
      <td>2.368750</td>
      <td>0.701080</td>
      <td>0.206157</td>
      <td>0.577836</td>
      <td>0.454153</td>
    </tr>
    <tr>
      <th>73</th>
      <td>(RefBks, DoItYBks)</td>
      <td>(ChildBks, CookBks)</td>
      <td>0.09250</td>
      <td>0.24200</td>
      <td>0.06125</td>
      <td>0.662162</td>
      <td>2.736207</td>
      <td>1.0</td>
      <td>0.038865</td>
      <td>2.243680</td>
      <td>0.699207</td>
      <td>0.224154</td>
      <td>0.554304</td>
      <td>0.457631</td>
    </tr>
    <tr>
      <th>60</th>
      <td>(YouthBks, DoItYBks)</td>
      <td>(ChildBks, CookBks)</td>
      <td>0.10325</td>
      <td>0.24200</td>
      <td>0.06700</td>
      <td>0.648910</td>
      <td>2.681448</td>
      <td>1.0</td>
      <td>0.042014</td>
      <td>2.158993</td>
      <td>0.699266</td>
      <td>0.240791</td>
      <td>0.536821</td>
      <td>0.462885</td>
    </tr>
    <tr>
      <th>80</th>
      <td>(GeogBks, RefBks)</td>
      <td>(ChildBks, CookBks)</td>
      <td>0.08175</td>
      <td>0.24200</td>
      <td>0.05025</td>
      <td>0.614679</td>
      <td>2.539995</td>
      <td>1.0</td>
      <td>0.030467</td>
      <td>1.967190</td>
      <td>0.660276</td>
      <td>0.183729</td>
      <td>0.491661</td>
      <td>0.411162</td>
    </tr>
    <tr>
      <th>69</th>
      <td>(GeogBks, YouthBks)</td>
      <td>(ChildBks, CookBks)</td>
      <td>0.10450</td>
      <td>0.24200</td>
      <td>0.06325</td>
      <td>0.605263</td>
      <td>2.501087</td>
      <td>1.0</td>
      <td>0.037961</td>
      <td>1.920267</td>
      <td>0.670211</td>
      <td>0.223301</td>
      <td>0.479239</td>
      <td>0.433313</td>
    </tr>
    <tr>
      <th>77</th>
      <td>(GeogBks, DoItYBks)</td>
      <td>(ChildBks, CookBks)</td>
      <td>0.10100</td>
      <td>0.24200</td>
      <td>0.06050</td>
      <td>0.599010</td>
      <td>2.475248</td>
      <td>1.0</td>
      <td>0.036058</td>
      <td>1.890321</td>
      <td>0.662959</td>
      <td>0.214159</td>
      <td>0.470989</td>
      <td>0.424505</td>
    </tr>
    <tr>
      <th>67</th>
      <td>(GeogBks, ChildBks, CookBks)</td>
      <td>(YouthBks)</td>
      <td>0.10950</td>
      <td>0.23825</td>
      <td>0.06325</td>
      <td>0.577626</td>
      <td>2.424452</td>
      <td>1.0</td>
      <td>0.037162</td>
      <td>1.803495</td>
      <td>0.659782</td>
      <td>0.222320</td>
      <td>0.445521</td>
      <td>0.421552</td>
    </tr>
    <tr>
      <th>70</th>
      <td>(ChildBks, RefBks, CookBks)</td>
      <td>(DoItYBks)</td>
      <td>0.10350</td>
      <td>0.25475</td>
      <td>0.06125</td>
      <td>0.591787</td>
      <td>2.323013</td>
      <td>1.0</td>
      <td>0.034883</td>
      <td>1.825642</td>
      <td>0.635276</td>
      <td>0.206229</td>
      <td>0.452247</td>
      <td>0.416110</td>
    </tr>
    <tr>
      <th>48</th>
      <td>(GeogBks, DoItYBks)</td>
      <td>(YouthBks)</td>
      <td>0.10100</td>
      <td>0.23825</td>
      <td>0.05450</td>
      <td>0.539604</td>
      <td>2.264864</td>
      <td>1.0</td>
      <td>0.030437</td>
      <td>1.654554</td>
      <td>0.621215</td>
      <td>0.191396</td>
      <td>0.395607</td>
      <td>0.384178</td>
    </tr>
    <tr>
      <th>63</th>
      <td>(ChildBks, RefBks, CookBks)</td>
      <td>(YouthBks)</td>
      <td>0.10350</td>
      <td>0.23825</td>
      <td>0.05525</td>
      <td>0.533816</td>
      <td>2.240573</td>
      <td>1.0</td>
      <td>0.030591</td>
      <td>1.634013</td>
      <td>0.617608</td>
      <td>0.192845</td>
      <td>0.388010</td>
      <td>0.382858</td>
    </tr>
  </tbody>
</table>
</div>



### 14.2 Collaborative Filtering

#### Example 3: Netflix Prize Contest


```python
random.seed(0)
nratings = 5000
randomData = pd.DataFrame({
    'itemID': [random.randint(0,99) for _ in range(nratings)],
    'userID': [random.randint(0,999) for _ in range(nratings)],
    'rating': [random.randint(1,5) for _ in range(nratings)]
})
def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    byUser = defaultdict(list)
    for p in predictions:
        byUser[p.uid].append(p)
    
    # For each user, reduce predictions to top-n
    for uid, userPredictions in byUser.items():
        byUser[uid] = heapq.nlargest(n, userPredictions, key=lambda p: p.est)
    return byUser
```


```python
# Convert the data set into the format required by the surprise package
# The columns must correspond to user id, item id, and ratings (in that order)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(randomData[['userID', 'itemID', 'rating']], reader)
# Split into training and test set
trainset, testset = train_test_split(data, test_size=.25, random_state=1)
## User-based filtering
# compute cosine similarity between users 
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)
# predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset) 
# Print the recommended items for each user
top_n = get_top_n(predictions, n=4)
print('Top-4 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
    print('User {}'.format(uid))
    for prediction in user_ratings:
        print('  Item {0.iid} ({0.est:.2f})'.format(prediction), end='')
    print()
```

    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    Top-4 recommended items for each user
    User 6
      Item 6 (5.00)  Item 77 (2.50)  Item 60 (1.00)
    User 222
      Item 77 (3.50)  Item 75 (2.78)
    User 424
      Item 14 (3.50)  Item 45 (3.10)  Item 54 (2.34)
    User 87
      Item 27 (3.00)  Item 54 (3.00)  Item 82 (3.00)  Item 32 (1.00)
    User 121
      Item 98 (3.48)  Item 32 (2.83)
    


```python
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)
# Predict rating for user 383 and item 7
algo.predict(383, 7)
```

    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    




    Prediction(uid=383, iid=7, r_ui=None, est=2.3661840936304324, details={'actual_k': 4, 'was_impossible': False})




```python

```
