## Chapter 15 Cluster Analysis

**Original Code Credit:**: Shmueli, Galit; Bruce, Peter C.; Gedeck, Peter; Patel, Nitin R.. Data Mining for Business Analytics Wiley.

*Modifications* have been made from the original textbook examples due to version changes in library dependencies and/or for clarity.

Download this notebook and data [**here**](https://github.com/dcyoung23/msba511/tree/main/resources/examples).

### Import Libraries


```python
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
```

### 15.1 Introduction

#### Example: Public Utilities


```python
utilities_df = pd.read_csv(os.path.join('data', 'Utilities.csv'))
# set row names to the utilities column
utilities_df.set_index('Company', inplace=True)
# while not required, the conversion of integer data to float
# will avoid a warning when applying the scale function
utilities_df = utilities_df.apply(lambda x: x.astype('float64'))
# compute Euclidean distance
d = pairwise.pairwise_distances(utilities_df, metric='euclidean')
pd.DataFrame(d, columns=utilities_df.index, index=utilities_df.index)
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
      <th>Company</th>
      <th>Arizona</th>
      <th>Boston</th>
      <th>Central</th>
      <th>Commonwealth</th>
      <th>NY</th>
      <th>Florida</th>
      <th>Hawaiian</th>
      <th>Idaho</th>
      <th>Kentucky</th>
      <th>Madison</th>
      <th>...</th>
      <th>Northern</th>
      <th>Oklahoma</th>
      <th>Pacific</th>
      <th>Puget</th>
      <th>San Diego</th>
      <th>Southern</th>
      <th>Texas</th>
      <th>Wisconsin</th>
      <th>United</th>
      <th>Virginia</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arizona</th>
      <td>0.000000</td>
      <td>3989.408076</td>
      <td>140.402855</td>
      <td>2654.277632</td>
      <td>5777.167672</td>
      <td>2050.529440</td>
      <td>1435.265019</td>
      <td>4006.104187</td>
      <td>671.276346</td>
      <td>2622.699002</td>
      <td>...</td>
      <td>1899.279821</td>
      <td>598.556633</td>
      <td>2609.045363</td>
      <td>6914.742065</td>
      <td>3363.061626</td>
      <td>1063.009074</td>
      <td>4430.251585</td>
      <td>1790.485648</td>
      <td>2427.588875</td>
      <td>1016.617691</td>
    </tr>
    <tr>
      <th>Boston</th>
      <td>3989.408076</td>
      <td>0.000000</td>
      <td>4125.044132</td>
      <td>1335.466502</td>
      <td>1788.068027</td>
      <td>6039.689076</td>
      <td>2554.287162</td>
      <td>7994.155985</td>
      <td>3318.276558</td>
      <td>1367.090634</td>
      <td>...</td>
      <td>2091.160485</td>
      <td>4586.302564</td>
      <td>1380.749962</td>
      <td>10903.146464</td>
      <td>629.760748</td>
      <td>5052.331669</td>
      <td>8419.610541</td>
      <td>2199.721665</td>
      <td>1562.210811</td>
      <td>5005.081262</td>
    </tr>
    <tr>
      <th>Central</th>
      <td>140.402855</td>
      <td>4125.044132</td>
      <td>0.000000</td>
      <td>2789.759674</td>
      <td>5912.552908</td>
      <td>1915.155154</td>
      <td>1571.295401</td>
      <td>3872.257626</td>
      <td>807.920792</td>
      <td>2758.559663</td>
      <td>...</td>
      <td>2035.441520</td>
      <td>461.341670</td>
      <td>2744.502847</td>
      <td>6780.430307</td>
      <td>3498.113013</td>
      <td>928.749249</td>
      <td>4295.014690</td>
      <td>1925.772564</td>
      <td>2563.637362</td>
      <td>883.535455</td>
    </tr>
    <tr>
      <th>Commonwealth</th>
      <td>2654.277632</td>
      <td>1335.466502</td>
      <td>2789.759674</td>
      <td>0.000000</td>
      <td>3123.153215</td>
      <td>4704.363099</td>
      <td>1219.560005</td>
      <td>6659.534567</td>
      <td>1983.314354</td>
      <td>43.648894</td>
      <td>...</td>
      <td>756.831954</td>
      <td>3250.984589</td>
      <td>56.644626</td>
      <td>9568.434429</td>
      <td>710.292965</td>
      <td>3717.202963</td>
      <td>7084.372839</td>
      <td>864.273153</td>
      <td>232.476871</td>
      <td>3670.018191</td>
    </tr>
    <tr>
      <th>NY</th>
      <td>5777.167672</td>
      <td>1788.068027</td>
      <td>5912.552908</td>
      <td>3123.153215</td>
      <td>0.000000</td>
      <td>7827.429211</td>
      <td>4342.093798</td>
      <td>9782.158178</td>
      <td>5106.094153</td>
      <td>3155.095594</td>
      <td>...</td>
      <td>3879.167462</td>
      <td>6373.743249</td>
      <td>3168.177463</td>
      <td>12691.155108</td>
      <td>2414.698757</td>
      <td>6840.150291</td>
      <td>10207.392630</td>
      <td>3987.335962</td>
      <td>3350.073118</td>
      <td>6793.035300</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>2050.529440</td>
      <td>6039.689076</td>
      <td>1915.155154</td>
      <td>4704.363099</td>
      <td>7827.429211</td>
      <td>0.000000</td>
      <td>3485.671562</td>
      <td>1959.731080</td>
      <td>2721.706296</td>
      <td>4672.829286</td>
      <td>...</td>
      <td>3949.092316</td>
      <td>1454.292604</td>
      <td>4659.356262</td>
      <td>4866.111649</td>
      <td>5413.093004</td>
      <td>988.044559</td>
      <td>2380.124974</td>
      <td>3840.227943</td>
      <td>4478.028874</td>
      <td>1035.981475</td>
    </tr>
    <tr>
      <th>Hawaiian</th>
      <td>1435.265019</td>
      <td>2554.287162</td>
      <td>1571.295401</td>
      <td>1219.560005</td>
      <td>4342.093798</td>
      <td>3485.671562</td>
      <td>0.000000</td>
      <td>5440.461781</td>
      <td>764.083188</td>
      <td>1187.941143</td>
      <td>...</td>
      <td>466.559118</td>
      <td>2032.614245</td>
      <td>1174.075616</td>
      <td>8349.366438</td>
      <td>1928.441480</td>
      <td>2498.149024</td>
      <td>5865.447190</td>
      <td>358.476293</td>
      <td>992.453252</td>
      <td>2451.185161</td>
    </tr>
    <tr>
      <th>Idaho</th>
      <td>4006.104187</td>
      <td>7994.155985</td>
      <td>3872.257626</td>
      <td>6659.534567</td>
      <td>9782.158178</td>
      <td>1959.731080</td>
      <td>5440.461781</td>
      <td>0.000000</td>
      <td>4676.638384</td>
      <td>6627.291780</td>
      <td>...</td>
      <td>5903.395450</td>
      <td>3412.263965</td>
      <td>6614.499239</td>
      <td>2909.014679</td>
      <td>7368.815437</td>
      <td>2943.535570</td>
      <td>447.828673</td>
      <td>5795.958815</td>
      <td>6432.132202</td>
      <td>2989.963982</td>
    </tr>
    <tr>
      <th>Kentucky</th>
      <td>671.276346</td>
      <td>3318.276558</td>
      <td>807.920792</td>
      <td>1983.314354</td>
      <td>5106.094153</td>
      <td>2721.706296</td>
      <td>764.083188</td>
      <td>4676.638384</td>
      <td>0.000000</td>
      <td>1951.628580</td>
      <td>...</td>
      <td>1228.436327</td>
      <td>1269.102099</td>
      <td>1938.026557</td>
      <td>7585.467294</td>
      <td>2692.212361</td>
      <td>1734.103297</td>
      <td>5101.414140</td>
      <td>1119.940014</td>
      <td>1756.378966</td>
      <td>1687.236030</td>
    </tr>
    <tr>
      <th>Madison</th>
      <td>2622.699002</td>
      <td>1367.090634</td>
      <td>2758.559663</td>
      <td>43.648894</td>
      <td>3155.095594</td>
      <td>4672.829286</td>
      <td>1187.941143</td>
      <td>6627.291780</td>
      <td>1951.628580</td>
      <td>0.000000</td>
      <td>...</td>
      <td>724.096182</td>
      <td>3219.825109</td>
      <td>53.301401</td>
      <td>9536.242192</td>
      <td>744.253668</td>
      <td>3685.510088</td>
      <td>7052.723883</td>
      <td>833.472995</td>
      <td>199.228400</td>
      <td>3638.097548</td>
    </tr>
    <tr>
      <th>Nevada</th>
      <td>8364.031051</td>
      <td>12353.062698</td>
      <td>8229.223281</td>
      <td>11018.057812</td>
      <td>14141.022579</td>
      <td>6314.359092</td>
      <td>9799.015552</td>
      <td>4359.599605</td>
      <td>9035.007488</td>
      <td>10986.098011</td>
      <td>...</td>
      <td>10262.157285</td>
      <td>7768.384793</td>
      <td>10973.010950</td>
      <td>1452.162005</td>
      <td>11727.066293</td>
      <td>7301.040864</td>
      <td>3934.617521</td>
      <td>10154.118793</td>
      <td>10791.049271</td>
      <td>7348.049019</td>
    </tr>
    <tr>
      <th>New England</th>
      <td>2923.136103</td>
      <td>1066.579432</td>
      <td>3058.707429</td>
      <td>271.452731</td>
      <td>2854.099482</td>
      <td>4973.506840</td>
      <td>1488.014909</td>
      <td>6928.326174</td>
      <td>2252.026717</td>
      <td>304.277034</td>
      <td>...</td>
      <td>1026.482994</td>
      <td>3519.977565</td>
      <td>314.354030</td>
      <td>9837.281834</td>
      <td>442.132760</td>
      <td>3986.102433</td>
      <td>7353.379146</td>
      <td>1134.145010</td>
      <td>496.687413</td>
      <td>3939.100355</td>
    </tr>
    <tr>
      <th>Northern</th>
      <td>1899.279821</td>
      <td>2091.160485</td>
      <td>2035.441520</td>
      <td>756.831954</td>
      <td>3879.167462</td>
      <td>3949.092316</td>
      <td>466.559118</td>
      <td>5903.395450</td>
      <td>1228.436327</td>
      <td>724.096182</td>
      <td>...</td>
      <td>0.000000</td>
      <td>2496.638890</td>
      <td>713.665046</td>
      <td>8812.303559</td>
      <td>1466.991954</td>
      <td>2961.834750</td>
      <td>6328.917948</td>
      <td>119.981262</td>
      <td>531.476328</td>
      <td>2914.204993</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <td>598.556633</td>
      <td>4586.302564</td>
      <td>461.341670</td>
      <td>3250.984589</td>
      <td>6373.743249</td>
      <td>1454.292604</td>
      <td>2032.614245</td>
      <td>3412.263965</td>
      <td>1269.102099</td>
      <td>3219.825109</td>
      <td>...</td>
      <td>2496.638890</td>
      <td>0.000000</td>
      <td>3205.748876</td>
      <td>6319.933836</td>
      <td>3959.240748</td>
      <td>470.164792</td>
      <td>3834.012257</td>
      <td>2386.942751</td>
      <td>3024.952355</td>
      <td>428.065259</td>
    </tr>
    <tr>
      <th>Pacific</th>
      <td>2609.045363</td>
      <td>1380.749962</td>
      <td>2744.502847</td>
      <td>56.644626</td>
      <td>3168.177463</td>
      <td>4659.356262</td>
      <td>1174.075616</td>
      <td>6614.499239</td>
      <td>1938.026557</td>
      <td>53.301401</td>
      <td>...</td>
      <td>713.665046</td>
      <td>3205.748876</td>
      <td>0.000000</td>
      <td>9523.413499</td>
      <td>754.612093</td>
      <td>3672.035402</td>
      <td>7039.262070</td>
      <td>820.164297</td>
      <td>186.388651</td>
      <td>3625.118869</td>
    </tr>
    <tr>
      <th>Puget</th>
      <td>6914.742065</td>
      <td>10903.146464</td>
      <td>6780.430307</td>
      <td>9568.434429</td>
      <td>12691.155108</td>
      <td>4866.111649</td>
      <td>8349.366438</td>
      <td>2909.014679</td>
      <td>7585.467294</td>
      <td>9536.242192</td>
      <td>...</td>
      <td>8812.303559</td>
      <td>6319.933836</td>
      <td>9523.413499</td>
      <td>0.000000</td>
      <td>10277.660378</td>
      <td>5851.893307</td>
      <td>2488.432223</td>
      <td>8704.721278</td>
      <td>9341.126615</td>
      <td>5898.576962</td>
    </tr>
    <tr>
      <th>San Diego</th>
      <td>3363.061626</td>
      <td>629.760748</td>
      <td>3498.113013</td>
      <td>710.292965</td>
      <td>2414.698757</td>
      <td>5413.093004</td>
      <td>1928.441480</td>
      <td>7368.815437</td>
      <td>2692.212361</td>
      <td>744.253668</td>
      <td>...</td>
      <td>1466.991954</td>
      <td>3959.240748</td>
      <td>754.612093</td>
      <td>10277.660378</td>
      <td>0.000000</td>
      <td>4426.041889</td>
      <td>7793.083947</td>
      <td>1573.408379</td>
      <td>938.522726</td>
      <td>4379.211818</td>
    </tr>
    <tr>
      <th>Southern</th>
      <td>1063.009074</td>
      <td>5052.331669</td>
      <td>928.749249</td>
      <td>3717.202963</td>
      <td>6840.150291</td>
      <td>988.044559</td>
      <td>2498.149024</td>
      <td>2943.535570</td>
      <td>1734.103297</td>
      <td>3685.510088</td>
      <td>...</td>
      <td>2961.834750</td>
      <td>470.164792</td>
      <td>3672.035402</td>
      <td>5851.893307</td>
      <td>4426.041889</td>
      <td>0.000000</td>
      <td>3367.318870</td>
      <td>2853.298778</td>
      <td>3490.422918</td>
      <td>59.325286</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>4430.251585</td>
      <td>8419.610541</td>
      <td>4295.014690</td>
      <td>7084.372839</td>
      <td>10207.392630</td>
      <td>2380.124974</td>
      <td>5865.447190</td>
      <td>447.828673</td>
      <td>5101.414140</td>
      <td>7052.723883</td>
      <td>...</td>
      <td>6328.917948</td>
      <td>3834.012257</td>
      <td>7039.262070</td>
      <td>2488.432223</td>
      <td>7793.083947</td>
      <td>3367.318870</td>
      <td>0.000000</td>
      <td>6220.296729</td>
      <td>6857.735864</td>
      <td>3414.831455</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <td>1790.485648</td>
      <td>2199.721665</td>
      <td>1925.772564</td>
      <td>864.273153</td>
      <td>3987.335962</td>
      <td>3840.227943</td>
      <td>358.476293</td>
      <td>5795.958815</td>
      <td>1119.940014</td>
      <td>833.472995</td>
      <td>...</td>
      <td>119.981262</td>
      <td>2386.942751</td>
      <td>820.164297</td>
      <td>8704.721278</td>
      <td>1573.408379</td>
      <td>2853.298778</td>
      <td>6220.296729</td>
      <td>0.000000</td>
      <td>640.786770</td>
      <td>2806.165712</td>
    </tr>
    <tr>
      <th>United</th>
      <td>2427.588875</td>
      <td>1562.210811</td>
      <td>2563.637362</td>
      <td>232.476871</td>
      <td>3350.073118</td>
      <td>4478.028874</td>
      <td>992.453252</td>
      <td>6432.132202</td>
      <td>1756.378966</td>
      <td>199.228400</td>
      <td>...</td>
      <td>531.476328</td>
      <td>3024.952355</td>
      <td>186.388651</td>
      <td>9341.126615</td>
      <td>938.522726</td>
      <td>3490.422918</td>
      <td>6857.735864</td>
      <td>640.786770</td>
      <td>0.000000</td>
      <td>3443.240967</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <td>1016.617691</td>
      <td>5005.081262</td>
      <td>883.535455</td>
      <td>3670.018191</td>
      <td>6793.035300</td>
      <td>1035.981475</td>
      <td>2451.185161</td>
      <td>2989.963982</td>
      <td>1687.236030</td>
      <td>3638.097548</td>
      <td>...</td>
      <td>2914.204993</td>
      <td>428.065259</td>
      <td>3625.118869</td>
      <td>5898.576962</td>
      <td>4379.211818</td>
      <td>59.325286</td>
      <td>3414.831455</td>
      <td>2806.165712</td>
      <td>3443.240967</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>




```python
# pandas uses sample standard deviation
utilities_df_norm = (utilities_df - utilities_df.mean())/utilities_df.std()
# compute normalized distance based on Sales and Fuel Cost
utilities_df_norm[['Sales', 'Fuel_Cost']]
d_norm = pairwise.pairwise_distances(utilities_df_norm[['Sales', 'Fuel_Cost']],
                                     metric='euclidean')
pd.DataFrame(d_norm, columns=utilities_df.index, index=utilities_df.index)
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
      <th>Company</th>
      <th>Arizona</th>
      <th>Boston</th>
      <th>Central</th>
      <th>Commonwealth</th>
      <th>NY</th>
      <th>Florida</th>
      <th>Hawaiian</th>
      <th>Idaho</th>
      <th>Kentucky</th>
      <th>Madison</th>
      <th>...</th>
      <th>Northern</th>
      <th>Oklahoma</th>
      <th>Pacific</th>
      <th>Puget</th>
      <th>San Diego</th>
      <th>Southern</th>
      <th>Texas</th>
      <th>Wisconsin</th>
      <th>United</th>
      <th>Virginia</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Arizona</th>
      <td>0.000000</td>
      <td>2.010329</td>
      <td>0.774179</td>
      <td>0.758738</td>
      <td>3.021907</td>
      <td>1.244422</td>
      <td>1.885248</td>
      <td>1.265638</td>
      <td>0.461292</td>
      <td>0.738650</td>
      <td>...</td>
      <td>0.564657</td>
      <td>0.182648</td>
      <td>1.570780</td>
      <td>1.947668</td>
      <td>2.509043</td>
      <td>0.913621</td>
      <td>1.247976</td>
      <td>0.521491</td>
      <td>2.761745</td>
      <td>1.252350</td>
    </tr>
    <tr>
      <th>Boston</th>
      <td>2.010329</td>
      <td>0.000000</td>
      <td>1.465703</td>
      <td>1.582821</td>
      <td>1.013370</td>
      <td>1.792397</td>
      <td>0.740283</td>
      <td>3.176654</td>
      <td>1.557738</td>
      <td>1.719632</td>
      <td>...</td>
      <td>1.940166</td>
      <td>2.166078</td>
      <td>0.478334</td>
      <td>3.501390</td>
      <td>0.679634</td>
      <td>1.634425</td>
      <td>2.890560</td>
      <td>1.654255</td>
      <td>1.100595</td>
      <td>1.479261</td>
    </tr>
    <tr>
      <th>Central</th>
      <td>0.774179</td>
      <td>1.465703</td>
      <td>0.000000</td>
      <td>1.015710</td>
      <td>2.432528</td>
      <td>0.631892</td>
      <td>1.156092</td>
      <td>1.732777</td>
      <td>0.419254</td>
      <td>1.102287</td>
      <td>...</td>
      <td>1.113433</td>
      <td>0.855093</td>
      <td>0.987772</td>
      <td>2.065643</td>
      <td>1.836762</td>
      <td>0.276440</td>
      <td>1.428159</td>
      <td>0.838967</td>
      <td>2.034824</td>
      <td>0.510365</td>
    </tr>
    <tr>
      <th>Commonwealth</th>
      <td>0.758738</td>
      <td>1.582821</td>
      <td>1.015710</td>
      <td>0.000000</td>
      <td>2.571969</td>
      <td>1.643857</td>
      <td>1.746027</td>
      <td>2.003230</td>
      <td>0.629994</td>
      <td>0.138758</td>
      <td>...</td>
      <td>0.377004</td>
      <td>0.937389</td>
      <td>1.258835</td>
      <td>2.699060</td>
      <td>2.202930</td>
      <td>1.278514</td>
      <td>1.998818</td>
      <td>0.243408</td>
      <td>2.547116</td>
      <td>1.502093</td>
    </tr>
    <tr>
      <th>NY</th>
      <td>3.021907</td>
      <td>1.013370</td>
      <td>2.432528</td>
      <td>2.571969</td>
      <td>0.000000</td>
      <td>2.635573</td>
      <td>1.411695</td>
      <td>4.162561</td>
      <td>2.566439</td>
      <td>2.705445</td>
      <td>...</td>
      <td>2.938637</td>
      <td>3.174588</td>
      <td>1.462019</td>
      <td>4.397433</td>
      <td>0.715629</td>
      <td>2.558409</td>
      <td>3.831132</td>
      <td>2.661786</td>
      <td>0.952507</td>
      <td>2.328691</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>1.244422</td>
      <td>1.792397</td>
      <td>0.631892</td>
      <td>1.643857</td>
      <td>2.635573</td>
      <td>0.000000</td>
      <td>1.228805</td>
      <td>1.764123</td>
      <td>1.025663</td>
      <td>1.722510</td>
      <td>...</td>
      <td>1.698624</td>
      <td>1.243634</td>
      <td>1.343185</td>
      <td>1.767581</td>
      <td>1.953423</td>
      <td>0.366744</td>
      <td>1.277920</td>
      <td>1.452417</td>
      <td>2.016493</td>
      <td>0.313847</td>
    </tr>
    <tr>
      <th>Hawaiian</th>
      <td>1.885248</td>
      <td>0.740283</td>
      <td>1.156092</td>
      <td>1.746027</td>
      <td>1.411695</td>
      <td>1.228805</td>
      <td>0.000000</td>
      <td>2.860189</td>
      <td>1.436822</td>
      <td>1.880361</td>
      <td>...</td>
      <td>2.027224</td>
      <td>1.997036</td>
      <td>0.560997</td>
      <td>2.995848</td>
      <td>0.726095</td>
      <td>1.205034</td>
      <td>2.463227</td>
      <td>1.711256</td>
      <td>0.879934</td>
      <td>0.929414</td>
    </tr>
    <tr>
      <th>Idaho</th>
      <td>1.265638</td>
      <td>3.176654</td>
      <td>1.732777</td>
      <td>2.003230</td>
      <td>4.162561</td>
      <td>1.764123</td>
      <td>2.860189</td>
      <td>0.000000</td>
      <td>1.650417</td>
      <td>1.950296</td>
      <td>...</td>
      <td>1.708409</td>
      <td>1.083449</td>
      <td>2.705579</td>
      <td>0.992092</td>
      <td>3.563727</td>
      <td>1.658671</td>
      <td>0.600089</td>
      <td>1.778813</td>
      <td>3.720421</td>
      <td>1.980715</td>
    </tr>
    <tr>
      <th>Kentucky</th>
      <td>0.461292</td>
      <td>1.557738</td>
      <td>0.419254</td>
      <td>0.629994</td>
      <td>2.566439</td>
      <td>1.025663</td>
      <td>1.436822</td>
      <td>1.650417</td>
      <td>0.000000</td>
      <td>0.697674</td>
      <td>...</td>
      <td>0.694524</td>
      <td>0.608401</td>
      <td>1.110854</td>
      <td>2.180496</td>
      <td>2.048098</td>
      <td>0.658996</td>
      <td>1.493274</td>
      <td>0.426780</td>
      <td>2.308613</td>
      <td>0.929141</td>
    </tr>
    <tr>
      <th>Madison</th>
      <td>0.738650</td>
      <td>1.719632</td>
      <td>1.102287</td>
      <td>0.138758</td>
      <td>2.705445</td>
      <td>1.722510</td>
      <td>1.880361</td>
      <td>1.950296</td>
      <td>0.697674</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.267198</td>
      <td>0.908665</td>
      <td>1.397240</td>
      <td>2.686215</td>
      <td>2.341644</td>
      <td>1.355786</td>
      <td>1.986625</td>
      <td>0.274061</td>
      <td>2.685340</td>
      <td>1.599587</td>
    </tr>
    <tr>
      <th>Nevada</th>
      <td>2.369479</td>
      <td>3.756513</td>
      <td>2.375975</td>
      <td>3.106084</td>
      <td>4.597006</td>
      <td>1.971518</td>
      <td>3.185311</td>
      <td>1.479526</td>
      <td>2.550689</td>
      <td>3.105627</td>
      <td>...</td>
      <td>2.923023</td>
      <td>2.211990</td>
      <td>3.293310</td>
      <td>0.487508</td>
      <td>3.899212</td>
      <td>2.145585</td>
      <td>1.133311</td>
      <td>2.862756</td>
      <td>3.887918</td>
      <td>2.284803</td>
    </tr>
    <tr>
      <th>New England</th>
      <td>2.425975</td>
      <td>0.684393</td>
      <td>1.737322</td>
      <td>2.153831</td>
      <td>0.846291</td>
      <td>1.831380</td>
      <td>0.608107</td>
      <td>3.458771</td>
      <td>1.966323</td>
      <td>2.292531</td>
      <td>...</td>
      <td>2.480456</td>
      <td>2.554109</td>
      <td>0.898094</td>
      <td>3.598846</td>
      <td>0.130663</td>
      <td>1.809354</td>
      <td>3.071178</td>
      <td>2.172473</td>
      <td>0.417866</td>
      <td>1.536436</td>
    </tr>
    <tr>
      <th>Northern</th>
      <td>0.564657</td>
      <td>1.940166</td>
      <td>1.113433</td>
      <td>0.377004</td>
      <td>2.938637</td>
      <td>1.698624</td>
      <td>2.027224</td>
      <td>1.708409</td>
      <td>0.694524</td>
      <td>0.267198</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.711050</td>
      <td>1.582591</td>
      <td>2.487892</td>
      <td>2.538720</td>
      <td>1.336887</td>
      <td>1.793287</td>
      <td>0.316160</td>
      <td>2.861293</td>
      <td>1.623614</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <td>0.182648</td>
      <td>2.166078</td>
      <td>0.855093</td>
      <td>0.937389</td>
      <td>3.174588</td>
      <td>1.243634</td>
      <td>1.997036</td>
      <td>1.083449</td>
      <td>0.608401</td>
      <td>0.908665</td>
      <td>...</td>
      <td>0.711050</td>
      <td>0.000000</td>
      <td>1.716739</td>
      <td>1.780656</td>
      <td>2.642155</td>
      <td>0.944295</td>
      <td>1.083449</td>
      <td>0.702684</td>
      <td>2.876646</td>
      <td>1.296548</td>
    </tr>
    <tr>
      <th>Pacific</th>
      <td>1.570780</td>
      <td>0.478334</td>
      <td>0.987772</td>
      <td>1.258835</td>
      <td>1.462019</td>
      <td>1.343185</td>
      <td>0.560997</td>
      <td>2.705579</td>
      <td>1.110854</td>
      <td>1.397240</td>
      <td>...</td>
      <td>1.582591</td>
      <td>1.716739</td>
      <td>0.000000</td>
      <td>3.027116</td>
      <td>0.958905</td>
      <td>1.160017</td>
      <td>2.412278</td>
      <td>1.276200</td>
      <td>1.288563</td>
      <td>1.035028</td>
    </tr>
    <tr>
      <th>Puget</th>
      <td>1.947668</td>
      <td>3.501390</td>
      <td>2.065643</td>
      <td>2.699060</td>
      <td>4.397433</td>
      <td>1.767581</td>
      <td>2.995848</td>
      <td>0.992092</td>
      <td>2.180496</td>
      <td>2.686215</td>
      <td>...</td>
      <td>2.487892</td>
      <td>1.780656</td>
      <td>3.027116</td>
      <td>0.000000</td>
      <td>3.720970</td>
      <td>1.867235</td>
      <td>0.700313</td>
      <td>2.456272</td>
      <td>3.763066</td>
      <td>2.069314</td>
    </tr>
    <tr>
      <th>San Diego</th>
      <td>2.509043</td>
      <td>0.679634</td>
      <td>1.836762</td>
      <td>2.202930</td>
      <td>0.715629</td>
      <td>1.953423</td>
      <td>0.726095</td>
      <td>3.563727</td>
      <td>2.048098</td>
      <td>2.341644</td>
      <td>...</td>
      <td>2.538720</td>
      <td>2.642155</td>
      <td>0.958905</td>
      <td>3.720970</td>
      <td>0.000000</td>
      <td>1.920035</td>
      <td>3.185942</td>
      <td>2.234632</td>
      <td>0.440163</td>
      <td>1.655498</td>
    </tr>
    <tr>
      <th>Southern</th>
      <td>0.913621</td>
      <td>1.634425</td>
      <td>0.276440</td>
      <td>1.278514</td>
      <td>2.558409</td>
      <td>0.366744</td>
      <td>1.205034</td>
      <td>1.658671</td>
      <td>0.658996</td>
      <td>1.355786</td>
      <td>...</td>
      <td>1.336887</td>
      <td>0.944295</td>
      <td>1.160017</td>
      <td>1.867235</td>
      <td>1.920035</td>
      <td>0.000000</td>
      <td>1.272784</td>
      <td>1.085774</td>
      <td>2.062067</td>
      <td>0.356298</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>1.247976</td>
      <td>2.890560</td>
      <td>1.428159</td>
      <td>1.998818</td>
      <td>3.831132</td>
      <td>1.277920</td>
      <td>2.463227</td>
      <td>0.600089</td>
      <td>1.493274</td>
      <td>1.986625</td>
      <td>...</td>
      <td>1.793287</td>
      <td>1.083449</td>
      <td>2.412278</td>
      <td>0.700313</td>
      <td>3.185942</td>
      <td>1.272784</td>
      <td>0.000000</td>
      <td>1.756136</td>
      <td>3.288460</td>
      <td>1.541576</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <td>0.521491</td>
      <td>1.654255</td>
      <td>0.838967</td>
      <td>0.243408</td>
      <td>2.661786</td>
      <td>1.452417</td>
      <td>1.711256</td>
      <td>1.778813</td>
      <td>0.426780</td>
      <td>0.274061</td>
      <td>...</td>
      <td>0.316160</td>
      <td>0.702684</td>
      <td>1.276200</td>
      <td>2.456272</td>
      <td>2.234632</td>
      <td>1.085774</td>
      <td>1.756136</td>
      <td>0.000000</td>
      <td>2.549040</td>
      <td>1.343306</td>
    </tr>
    <tr>
      <th>United</th>
      <td>2.761745</td>
      <td>1.100595</td>
      <td>2.034824</td>
      <td>2.547116</td>
      <td>0.952507</td>
      <td>2.016493</td>
      <td>0.879934</td>
      <td>3.720421</td>
      <td>2.308613</td>
      <td>2.685340</td>
      <td>...</td>
      <td>2.861293</td>
      <td>2.876646</td>
      <td>1.288563</td>
      <td>3.763066</td>
      <td>0.440163</td>
      <td>2.062067</td>
      <td>3.288460</td>
      <td>2.549040</td>
      <td>0.000000</td>
      <td>1.749930</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <td>1.252350</td>
      <td>1.479261</td>
      <td>0.510365</td>
      <td>1.502093</td>
      <td>2.328691</td>
      <td>0.313847</td>
      <td>0.929414</td>
      <td>1.980715</td>
      <td>0.929141</td>
      <td>1.599587</td>
      <td>...</td>
      <td>1.623614</td>
      <td>1.296548</td>
      <td>1.035028</td>
      <td>2.069314</td>
      <td>1.655498</td>
      <td>0.356298</td>
      <td>1.541576</td>
      <td>1.343306</td>
      <td>1.749930</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>



### 15.4 Hierarchical (Agglomerative) Clustering

**We will focus on kMeans Clustering but code examples for hierarchical clustering are provided for example purposes.**


```python
fig, axes = plt.subplots(2, 1, figsize=(10,10))
# in linkage() set argument method =
# 'single', 'complete', 'average', 'weighted', centroid', 'median', 'ward'
Z = linkage(utilities_df_norm, method='single')
ax1 = axes[0]
dendrogram(Z, labels=utilities_df_norm.index, color_threshold=2.75, ax=ax1)
ax1.set_title('Hierarchical Clustering Dendrogram (Single Linkage)')
Z = linkage(utilities_df_norm, method='average')
ax2 = axes[1]
dendrogram(Z, labels=utilities_df_norm.index, color_threshold=3.6, ax=ax2)
ax2.set_title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.tight_layout()
plt.show()
```


    
![png](assets\ch15_output_9_0.png)
    



```python
memb = fcluster(linkage(utilities_df_norm, method='single'), 6, criterion='maxclust')
memb = pd.Series(memb, index=utilities_df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))
```

    1 :  Idaho, Puget
    2 :  Arizona , Boston , Commonwealth, Florida , Hawaiian , Kentucky, Madison , New England, Northern, Oklahoma, Pacific , Southern, Texas, Wisconsin, United, Virginia
    3 :  Central 
    4 :  San Diego
    5 :  Nevada
    6 :  NY
    


```python
memb = fcluster(linkage(utilities_df_norm, method='average'), 6, criterion='maxclust')
memb = pd.Series(memb, index=utilities_df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))
```

    1 :  Idaho, Nevada, Puget
    2 :  Hawaiian , New England, Pacific , United
    3 :  San Diego
    4 :  Boston , Commonwealth, Madison , Northern, Wisconsin, Virginia
    5 :  Arizona , Central , Florida , Kentucky, Oklahoma, Southern, Texas
    6 :  NY
    


```python
# set labels as cluster membership and utility name
utilities_df_norm.index = ['{}: {}'.format(cluster, state)
                           for cluster, state in zip(memb, utilities_df_norm.index)]
# plot heatmap
# the '_r' suffix reverses the color mapping to large = dark
sns.clustermap(utilities_df_norm, method='average', col_cluster=False, cmap='mako_r')
plt.show()
```


    
![png](assets\ch15_output_12_0.png)
    


### 15.5 Non-Hierarchical Clustering: The k-Means Algorithm



```python
# Normalize distances
utilities_df_norm = utilities_df.apply(preprocessing.scale, axis=0)
kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(utilities_df_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=utilities_df_norm.index)
for key, item in memb.groupby(memb):
     print(key, ': ', ', '.join(item.index))
```

    0 :  Idaho, Puget
    1 :  Arizona , Central , Florida , Kentucky, Oklahoma, Southern, Texas
    2 :  Commonwealth, Madison , Northern, Wisconsin, Virginia
    3 :  Boston , Hawaiian , New England, Pacific , San Diego, United
    4 :  Nevada
    5 :  NY
    


```python
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=utilities_df_norm.columns)
centroids
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
      <th>Fixed_charge</th>
      <th>RoR</th>
      <th>Cost</th>
      <th>Load_factor</th>
      <th>Demand_growth</th>
      <th>Sales</th>
      <th>Nuclear</th>
      <th>Fuel_Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.088252</td>
      <td>-0.541112</td>
      <td>1.995766</td>
      <td>-0.109502</td>
      <td>0.987702</td>
      <td>1.621068</td>
      <td>-0.731447</td>
      <td>-1.174696</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.516184</td>
      <td>0.797896</td>
      <td>-1.009097</td>
      <td>-0.345490</td>
      <td>-0.501098</td>
      <td>0.360140</td>
      <td>-0.535523</td>
      <td>-0.420198</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.011599</td>
      <td>0.339180</td>
      <td>0.224086</td>
      <td>-0.366466</td>
      <td>0.170386</td>
      <td>-0.411331</td>
      <td>1.601868</td>
      <td>-0.609460</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.632893</td>
      <td>-0.639936</td>
      <td>0.206692</td>
      <td>1.175321</td>
      <td>0.057691</td>
      <td>-0.757719</td>
      <td>-0.380962</td>
      <td>1.203616</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.019709</td>
      <td>-1.476137</td>
      <td>0.119723</td>
      <td>-1.256665</td>
      <td>1.069762</td>
      <td>2.458495</td>
      <td>-0.731447</td>
      <td>-0.616086</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.085268</td>
      <td>-0.883194</td>
      <td>0.591840</td>
      <td>-1.325495</td>
      <td>-0.735555</td>
      <td>-1.618644</td>
      <td>0.219434</td>
      <td>1.732470</td>
    </tr>
  </tbody>
</table>
</div>




```python
# calculate the distances of each data point to the cluster centers
distances = kmeans.transform(utilities_df_norm)
# find closest cluster for each data point
minSquaredDistances = distances.min(axis=1) ** 2
# combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 'cluster': kmeans.labels_},
    index=utilities_df_norm.index)
# group by cluster and print information
for cluster, data in df.groupby('cluster'):
    count = len(data)
    withinClustSS = data.squaredDistance.sum()
    print(f'Cluster {cluster} ({count} members): {withinClustSS:.2f} within cluster ')

```

    Cluster 0 (2 members): 2.54 within cluster 
    Cluster 1 (7 members): 27.77 within cluster 
    Cluster 2 (5 members): 10.66 within cluster 
    Cluster 3 (6 members): 22.20 within cluster 
    Cluster 4 (1 members): 0.00 within cluster 
    Cluster 5 (1 members): 0.00 within cluster 
    


```python
centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
plt.figure(figsize=(10,6))
parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
```


    
![png](assets\ch15_output_17_0.png)
    



```python
inertia = []
for n_clusters in range(1, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(utilities_df_norm)
    inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(1, 7), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.show()
```


    
![png](assets\ch15_output_18_0.png)
    



```python

```
