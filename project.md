<font size="+3"><strong>5.5. Bankruptcy in Taiwan 🇹🇼</strong></font>


```python
# Import libraries here
import gzip
import json
import pickle

import ipywidgets as widgets
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from IPython.display import VimeoVideo
from ipywidgets import interact
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from teaching_tools.widgets import ConfusionMatrixWidget
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
```

# Prepare Data

## Import

**Task 5.5.1**


```python
# Load data file
with gzip.open("data/taiwan-bankruptcy-data.json.gz", "rb") as f:
    taiwan_data = json.load(f)

print(type(taiwan_data))
```

    <class 'dict'>


**Task 5.5.2**

<div class="alert alert-info" role="alert">
    <b>Tip:</b> The data in this assignment might be organized differently than the data from the project, so be sure to inspect it first. 
</div>


```python
taiwan_data_keys = taiwan_data.keys()
print(taiwan_data_keys)
```

    dict_keys(['schema', 'metadata', 'observations'])


**Task 5.5.3**


```python
n_companies = len(taiwan_data["observations"])
print(n_companies)
```

    6137


**Task 5.5.4**


```python
n_features = len(list(taiwan_data["observations"][0].keys()))
print(n_features)
```

    97


**Task 5.5.5**


```python
# Create wrangle function
def wrangle(filepath):
    with gzip.open(filepath, "rb") as f:
        taiwan_data = json.load(f)
        df = pd.DataFrame().from_dict(taiwan_data["observations"]).set_index("id")
    return df

```


```python
df = wrangle("data/taiwan-bankruptcy-data.json.gz")
print("df shape:", df.shape)
df.head()
```

    df shape: (6137, 96)





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
      <th>bankrupt</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>...</th>
      <th>feat_86</th>
      <th>feat_87</th>
      <th>feat_88</th>
      <th>feat_89</th>
      <th>feat_90</th>
      <th>feat_91</th>
      <th>feat_92</th>
      <th>feat_93</th>
      <th>feat_94</th>
      <th>feat_95</th>
    </tr>
    <tr>
      <th>id</th>
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
      <th>1</th>
      <td>True</td>
      <td>0.370594</td>
      <td>0.424389</td>
      <td>0.405750</td>
      <td>0.601457</td>
      <td>0.601457</td>
      <td>0.998969</td>
      <td>0.796887</td>
      <td>0.808809</td>
      <td>0.302646</td>
      <td>...</td>
      <td>0.716845</td>
      <td>0.009219</td>
      <td>0.622879</td>
      <td>0.601453</td>
      <td>0.827890</td>
      <td>0.290202</td>
      <td>0.026601</td>
      <td>0.564050</td>
      <td>1</td>
      <td>0.016469</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>0.464291</td>
      <td>0.538214</td>
      <td>0.516730</td>
      <td>0.610235</td>
      <td>0.610235</td>
      <td>0.998946</td>
      <td>0.797380</td>
      <td>0.809301</td>
      <td>0.303556</td>
      <td>...</td>
      <td>0.795297</td>
      <td>0.008323</td>
      <td>0.623652</td>
      <td>0.610237</td>
      <td>0.839969</td>
      <td>0.283846</td>
      <td>0.264577</td>
      <td>0.570175</td>
      <td>1</td>
      <td>0.020794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>0.426071</td>
      <td>0.499019</td>
      <td>0.472295</td>
      <td>0.601450</td>
      <td>0.601364</td>
      <td>0.998857</td>
      <td>0.796403</td>
      <td>0.808388</td>
      <td>0.302035</td>
      <td>...</td>
      <td>0.774670</td>
      <td>0.040003</td>
      <td>0.623841</td>
      <td>0.601449</td>
      <td>0.836774</td>
      <td>0.290189</td>
      <td>0.026555</td>
      <td>0.563706</td>
      <td>1</td>
      <td>0.016474</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>0.399844</td>
      <td>0.451265</td>
      <td>0.457733</td>
      <td>0.583541</td>
      <td>0.583541</td>
      <td>0.998700</td>
      <td>0.796967</td>
      <td>0.808966</td>
      <td>0.303350</td>
      <td>...</td>
      <td>0.739555</td>
      <td>0.003252</td>
      <td>0.622929</td>
      <td>0.583538</td>
      <td>0.834697</td>
      <td>0.281721</td>
      <td>0.026697</td>
      <td>0.564663</td>
      <td>1</td>
      <td>0.023982</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>0.465022</td>
      <td>0.538432</td>
      <td>0.522298</td>
      <td>0.598783</td>
      <td>0.598783</td>
      <td>0.998973</td>
      <td>0.797366</td>
      <td>0.809304</td>
      <td>0.303475</td>
      <td>...</td>
      <td>0.795016</td>
      <td>0.003878</td>
      <td>0.623521</td>
      <td>0.598782</td>
      <td>0.839973</td>
      <td>0.278514</td>
      <td>0.024752</td>
      <td>0.575617</td>
      <td>1</td>
      <td>0.035490</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 96 columns</p>
</div>



## Explore

**Task 5.5.6**


```python
nans_by_col = pd.Series({col: df[col].isna().sum() for col in df.columns})
print("nans_by_col shape:", nans_by_col.shape)
nans_by_col.head()
```

    nans_by_col shape: (96,)





    bankrupt    0
    feat_1      0
    feat_2      0
    feat_3      0
    feat_4      0
    dtype: int64



<div class="alert alert-info" role="alert">
  <strong>Slight Code Change</strong>

In the following task, you'll notice a small change in how plots are created compared to what you saw in the lessons.
While the lessons use the global matplotlib method like <code>plt.plot(...)</code>, in this task, you are expected to use the object-oriented (OOP) API instead.
This means creating your plots using <code>fig, ax = plt.subplots()</code> and then calling plotting methods on the <code>ax</code> object, such as <code>ax.plot(...)</code>, <code>ax.hist(...)</code>, or <code>ax.scatter(...)</code>.

If you're using pandas’ or seaborn’s built-in plotting methods (like <code>df.plot()</code> or <code>sns.lineplot()</code>), make sure to pass the <code>ax=ax</code> argument so that the plot is rendered on the correct axes.

This approach is considered best practice and will be used consistently across all graded tasks that involve matplotlib.
</div>


**Task 5.5.7**


```python
# Plot class balance
fig, ax = plt.subplots()
df["bankrupt"].value_counts(normalize=True).plot(kind="bar", ax=ax)
ax.set_title("Class Balance")
ax.set_xlabel("Bankrupt")
ax.set_ylabel("Frequency")
plt.show()
```


    
![png](output_20_0.png)
    


## Split

**Task 5.5.8**


```python
target = "bankrupt"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)
```

    X shape: (6137, 95)
    y shape: (6137,)


**Task 5.5.9**


```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
```

    X_train shape: (4909, 95)
    y_train shape: (4909,)
    X_test shape: (1228, 95)
    y_test shape: (1228,)


## Resample

**Task 5.5.10**


```python
over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train,y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()
```

    X_train_over shape: (9512, 95)





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
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>feat_10</th>
      <th>...</th>
      <th>feat_86</th>
      <th>feat_87</th>
      <th>feat_88</th>
      <th>feat_89</th>
      <th>feat_90</th>
      <th>feat_91</th>
      <th>feat_92</th>
      <th>feat_93</th>
      <th>feat_94</th>
      <th>feat_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.535855</td>
      <td>0.599160</td>
      <td>0.594411</td>
      <td>0.627099</td>
      <td>0.627099</td>
      <td>0.999220</td>
      <td>0.797686</td>
      <td>0.809591</td>
      <td>0.303518</td>
      <td>0.781865</td>
      <td>...</td>
      <td>0.834091</td>
      <td>0.022025</td>
      <td>0.624364</td>
      <td>0.627101</td>
      <td>0.841977</td>
      <td>0.275384</td>
      <td>0.026791</td>
      <td>0.565158</td>
      <td>1</td>
      <td>0.147943</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.554136</td>
      <td>0.612734</td>
      <td>0.595000</td>
      <td>0.607388</td>
      <td>0.607388</td>
      <td>0.999120</td>
      <td>0.797614</td>
      <td>0.809483</td>
      <td>0.303600</td>
      <td>0.781754</td>
      <td>...</td>
      <td>0.840293</td>
      <td>0.002407</td>
      <td>0.624548</td>
      <td>0.607385</td>
      <td>0.842645</td>
      <td>0.276532</td>
      <td>0.026791</td>
      <td>0.565158</td>
      <td>1</td>
      <td>0.062544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.549554</td>
      <td>0.603467</td>
      <td>0.599122</td>
      <td>0.620166</td>
      <td>0.620166</td>
      <td>0.999119</td>
      <td>0.797569</td>
      <td>0.809470</td>
      <td>0.303524</td>
      <td>0.781740</td>
      <td>...</td>
      <td>0.840403</td>
      <td>0.000840</td>
      <td>0.624010</td>
      <td>0.620163</td>
      <td>0.842873</td>
      <td>0.277249</td>
      <td>0.026800</td>
      <td>0.565200</td>
      <td>1</td>
      <td>0.047929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.543801</td>
      <td>0.603249</td>
      <td>0.606992</td>
      <td>0.622515</td>
      <td>0.622515</td>
      <td>0.999259</td>
      <td>0.797728</td>
      <td>0.809649</td>
      <td>0.303510</td>
      <td>0.781930</td>
      <td>...</td>
      <td>0.831514</td>
      <td>0.006176</td>
      <td>0.626775</td>
      <td>0.622513</td>
      <td>0.842989</td>
      <td>0.280013</td>
      <td>0.026839</td>
      <td>0.565375</td>
      <td>1</td>
      <td>0.028386</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.498659</td>
      <td>0.562364</td>
      <td>0.546978</td>
      <td>0.603670</td>
      <td>0.603670</td>
      <td>0.998904</td>
      <td>0.797584</td>
      <td>0.809459</td>
      <td>0.304000</td>
      <td>0.781713</td>
      <td>...</td>
      <td>0.811988</td>
      <td>0.004256</td>
      <td>0.623674</td>
      <td>0.603669</td>
      <td>0.841105</td>
      <td>0.277628</td>
      <td>0.026897</td>
      <td>0.565618</td>
      <td>1</td>
      <td>0.043080</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 95 columns</p>
</div>



# Build Model

## Iterate

**Task 5.5.11**


```python
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_over, y_train_over)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>



**Task 5.5.12**

<div class="alert alert-info" role="alert">
    <p><b>Tip:</b> Use your CV scores to evaluate different classifiers. Choose the one that gives you the best scores.</p>
</div>


```python
cv_scores = cross_val_score(
    clf,
    X_train_over,
    y_train_over,
    cv=5,
    n_jobs=-1
)
print(cv_scores)
```

    [0.99316868 0.99474514 0.99369085 0.99369085 0.9957939 ]



```python
# Run this code cell

params = {
    "max_depth": range(30, 50, 10),
    "n_estimators": range(25, 51, 25),
}
```

**Task 5.5.13**


```python
model = GridSearchCV(
    estimator= clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
```

**Ungraded Task:** Fit your model to the over-sampled training data. 


```python
model.fit(X_train_over, y_train_over)
```

    Fitting 5 folds for each of 4 candidates, totalling 20 fits





<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: range(30, 50, 10),
                         &#x27;n_estimators&#x27;: range(25, 51, 25)},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: range(30, 50, 10),
                         &#x27;n_estimators&#x27;: range(25, 51, 25)},
             verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>



**Task 5.5.14**


```python
cv_results = pd.DataFrame(model.cv_results_)
cv_results.head(5)
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_depth</th>
      <th>param_n_estimators</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.700305</td>
      <td>0.016307</td>
      <td>0.008294</td>
      <td>0.000947</td>
      <td>30</td>
      <td>25</td>
      <td>{'max_depth': 30, 'n_estimators': 25}</td>
      <td>0.992643</td>
      <td>0.993694</td>
      <td>0.992114</td>
      <td>0.993165</td>
      <td>0.993165</td>
      <td>0.992956</td>
      <td>0.000537</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.436877</td>
      <td>0.042596</td>
      <td>0.013953</td>
      <td>0.002285</td>
      <td>30</td>
      <td>50</td>
      <td>{'max_depth': 30, 'n_estimators': 50}</td>
      <td>0.992118</td>
      <td>0.993694</td>
      <td>0.992114</td>
      <td>0.993165</td>
      <td>0.994217</td>
      <td>0.993061</td>
      <td>0.000841</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.707943</td>
      <td>0.014181</td>
      <td>0.008914</td>
      <td>0.002172</td>
      <td>40</td>
      <td>25</td>
      <td>{'max_depth': 40, 'n_estimators': 25}</td>
      <td>0.994220</td>
      <td>0.994220</td>
      <td>0.994742</td>
      <td>0.993165</td>
      <td>0.994742</td>
      <td>0.994218</td>
      <td>0.000576</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.415400</td>
      <td>0.037697</td>
      <td>0.012732</td>
      <td>0.000074</td>
      <td>40</td>
      <td>50</td>
      <td>{'max_depth': 40, 'n_estimators': 50}</td>
      <td>0.992643</td>
      <td>0.995271</td>
      <td>0.994217</td>
      <td>0.993691</td>
      <td>0.996320</td>
      <td>0.994428</td>
      <td>0.001270</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Task 5.5.15**


```python
best_params = model.best_params_
print(best_params)
```

    {'max_depth': 40, 'n_estimators': 50}


## Evaluate

**Ungraded Task:** Test the quality of your model by calculating accuracy scores for the training and test data.


```python
acc_train = model.score(X_train,y_train)
acc_test = model.score(X_test, y_test)

print("Model Training Accuracy:", round(acc_train, 4))
print("Model Test Accuracy:", round(acc_test, 4))
```

    Model Training Accuracy: 1.0
    Model Test Accuracy: 0.9772


**Task 5.5.16**


```python
# write code here
confusionmatrix = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
dis = ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=model.classes_)
dis.plot(ax=ax, colorbar=True)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x77b4456752d0>




    
![png](output_49_1.png)
    


**Task 5.5.17**


```python
class_report = classification_report(y_test, model.predict(X_test))
print(class_report)
```

                  precision    recall  f1-score   support
    
           False       0.98      0.99      0.99      1191
            True       0.71      0.41      0.52        37
    
        accuracy                           0.98      1228
       macro avg       0.85      0.70      0.75      1228
    weighted avg       0.97      0.98      0.97      1228
    


# Communicate

**Task 5.5.18**


```python
features = X_test.columns

importances = model.best_estimator_.feature_importances_

feat_imp = pd.Series(importances, index=features).sort_values()

# Select the same top 10 (keeps original order)
top10 = feat_imp.tail(10)

# Plot using fig, ax
fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(top10.index, top10.values)
ax.set_xlabel("Gini Importance")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance")

plt.tight_layout()
plt.show()


```


    
![png](output_54_0.png)
    


**Task 5.5.19**


```python
# Save model
with open("model-5-5.pkl", "wb") as f:
    pickle.dump(model, f)
```

**Task 5.5.20**


```python
# Import your module
from my_predictor_assignment import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
    model_filepath="model-5-5.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()
```

    predictions shape: (682,)





    id
    18    False
    20    False
    24    False
    32    False
    38    False
    Name: bankrupt, dtype: bool



<div class="alert alert-info" role="alert">
    <b>Tip:</b> If you get an <code style="color:#E45E5C;background-color:#FEDDDE">ImportError</code> when you try to import <code>make_predictions</code> from <code>my_predictor_assignment</code>, try restarting your kernel. Go to the <b>Kernel</b> menu and click on  <b>Restart Kernel and Clear All Outputs</b>. Then rerun just the cell above. ☝️
</div>

---
Copyright 2023 WorldQuant University. This
content is licensed solely for personal use. Redistribution or
publication of this material is strictly prohibited.

