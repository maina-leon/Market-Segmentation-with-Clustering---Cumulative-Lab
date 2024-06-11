
# Market Segmentation with Clustering - Lab

## Introduction

In this lab, you'll use your knowledge of clustering to perform market segmentation on a real-world dataset!

## Objectives

In this lab you will: 

- Use clustering to create and interpret market segmentation on real-world data 

## Getting Started

In this lab, you're going to work with the [Wholesale customers dataset](https://archive.ics.uci.edu/ml/datasets/wholesale+customers) from the UCI Machine Learning datasets repository. This dataset contains data on wholesale purchasing information from real businesses. These businesses range from small cafes and hotels to grocery stores and other retailers. 

Here's the data dictionary for this dataset:

|      Column      |                                               Description                                              |
|:----------------:|:------------------------------------------------------------------------------------------------------:|
|       FRESH      |                    Annual spending on fresh products, such as fruits and vegetables                    |
|       MILK       |                               Annual spending on milk and dairy products                               |
|      GROCERY     |                                   Annual spending on grocery products                                  |
|      FROZEN      |                                   Annual spending on frozen products                                   |
| DETERGENTS_PAPER |                  Annual spending on detergents, cleaning supplies, and paper products                  |
|   DELICATESSEN   |                           Annual spending on meats and delicatessen products                           |
|      CHANNEL     | Type of customer.  1=Hotel/Restaurant/Cafe, 2=Retailer. (This is what we'll use clustering to predict) |
|      REGION      |            Region of Portugal that the customer is located in. (This column will be dropped)           |



One benefit of working with this dataset for practice with segmentation is that we actually have the ground-truth labels of what market segment each customer actually belongs to. For this reason, we'll borrow some methodology from supervised learning and store these labels separately, so that we can use them afterward to check how well our clustering segmentation actually performed. 

Let's get started by importing everything we'll need.

In the cell below:

* Import `pandas`, `numpy`, and `matplotlib.pyplot`, and set the standard alias for each. 
* Use `numpy` to set a random seed of `0`.
* Set all matplotlib visualizations to appear inline.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
%matplotlib inline

```

Now, let's load our data and inspect it. You'll find the data stored in `'wholesale_customers_data.csv'`. 

In the cell below, load the data into a DataFrame and then display the first five rows to ensure everything loaded correctly.


```python
raw_df = pd.read_csv('Wholesale customers data.csv')
raw_df.head()
```

Now, let's go ahead and store the `'Channel'` column in a separate variable and then drop both the `'Channel'` and `'Region'` columns. Then, display the first five rows of the new DataFrame to ensure everything worked correctly. 


```python
channels = raw_df['Channel']
df = raw_df.drop(columns=['Channel', 'Region'])
df.head()
```

Now, let's get right down to it and begin our clustering analysis. 

In the cell below:

* Import `KMeans` from `sklearn.cluster`, and then create an instance of it. Set the number of clusters to `2`
* Fit it to the data (`df`) 
* Get the predictions from the clustering algorithm and store them in `cluster_preds` 


```python
from sklearn.cluster import KMeans
```


```python
k_means = KMeans(n_clusters=2, random_state=0)
k_means.fit(df)
cluster_preds = k_means.predict(df)
```

Now, use some of the metrics to check the performance. You'll use `calinski_harabasz_score()` and `adjusted_rand_score()`, which can both be found inside [`sklearn.metrics`](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation). 

In the cell below, import these scoring functions. 


```python
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
```

Now, start with CH score to get the variance ratio. 


```python
# Calinski-Harabasz score
ch_score = calinski_harabasz_score(df, cluster_preds)

# Display the Calinski-Harabasz score
print(f"Calinski-Harabasz Score: {ch_score}")

```

Although you don't have any other numbers to compare this to, this is a pretty low score, suggesting that the clusters aren't great. 

Since you actually have ground-truth labels, in this case you can use `adjusted_rand_score()` to check how well the clustering performed. Adjusted Rand score is meant to compare two clusterings, which the score can interpret our labels as. This will tell us how similar the predicted clusters are to the actual channels. 

Adjusted Rand score is bounded between -1 and 1. A score close to 1 shows that the clusters are almost identical. A score close to 0 means that predictions are essentially random, while a score close to -1 means that the predictions are pathologically bad, since they are worse than random chance. 

In the cell below, call `adjusted_rand_score()` and pass in `channels` and `cluster_preds` to see how well your first iteration of clustering performed. 


```python
# Adjusted Rand Index
ari_score = adjusted_rand_score(channels, cluster_preds)

print(f"Adjusted Rand Index: {ari_score}")

```

According to these results, the clusterings were essentially no better than random chance. Let's see if you can improve this. 

### Scaling our dataset

Recall that k-means clustering is heavily affected by scaling. Since the clustering algorithm is distance-based, this makes sense. Let's use `StandardScaler` to scale our dataset and then try our clustering again and see if the results are different. 

In the cells below:

* Import and instantiate [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and use it to transform the dataset  
* Instantiate and fit k-means to this scaled data, and then use it to predict clusters 
* Calculate the adjusted Rand score for these new predictions 


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
```


```python
scaled_k_means =  KMeans(n_clusters=2, random_state=0)
scaled_k_means.fit(scaled_df)
scaled_preds = scaled_k_means.predict(scaled_df)
```


```python
ars = adjusted_rand_score(channels, scaled_preds)
print(f'Adjusted Rand Score (Scaled Data): {ars}')
```

That's a big improvement! Although it's not perfect, we can see that scaling our data had a significant effect on the quality of our clusters. 

## Incorporating PCA

Since clustering algorithms are distance-based, this means that dimensionality has a definite effect on their performance. The greater the dimensionality of the dataset, the greater the total area that we have to worry about our clusters existing in. Let's try using Principal Component Analysis to transform our data and see if this affects the performance of our clustering algorithm. 

Since you've already seen PCA in a previous section, we will let you figure this out by yourself. 

In the cells below:

* Import [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) from the appropriate module in sklearn 
* Create a `PCA` instance and use it to transform our scaled data  
* Investigate the explained variance ratio for each Principal Component. Consider dropping certain components to reduce dimensionality if you feel it is worth the loss of information 
* Create a new `KMeans` object, fit it to our PCA-transformed data, and check the adjusted Rand score of the predictions it makes. 

**_NOTE:_** Your overall goal here is to get the highest possible adjusted Rand score. Don't be afraid to change parameters and rerun things to see how it changes. 


```python
from sklearn.decomposition import PCA
```


```python
# Instantiate PCA
pca = PCA()

# Fit PCA to the scaled data and transform it
pca_data = pca.fit_transform(scaled_df)

# Investigate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Display the explained variance ratio for each component
print(f"Explained Variance Ratio: {explained_variance_ratio}")

# Plot the cumulative explained variance to decide on the number of components
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

```


```python
# Number of components to retain 
n_components = 4

# Instantiate PCA with the chosen number of components
pca = PCA(n_components=n_components)

# Fit PCA to the scaled data and transform it
pca_data_reduced = pca.fit_transform(scaled_df)

```


```python
# Instantiate KMeans 
kmeans_pca = KMeans(n_clusters=2, random_state=0)

# Fit the KMeans model to the PCA-transformed data
kmeans_pca.fit(pca_data_reduced)

# Get the predictions from the clustering algorithm
cluster_preds_pca = kmeans_pca.predict(pca_data_reduced)

# Calculate the Adjusted Rand Index for the PCA-transformed data
ari_score_pca = adjusted_rand_score(channels, cluster_preds_pca)

# Display the Adjusted Rand Index for the PCA-transformed data
print(f"Adjusted Rand Index (PCA-Transformed Data): {ari_score_pca}")

```
**_Question_**:  What was the Highest Adjusted Rand Score you achieved? Interpret this score and determine the overall quality of the clustering. Did PCA affect the performance overall?  How many principal components resulted in the best overall clustering performance? Why do you think this is?

Write your answer below this line:
_______________________________________________________________________________________________________________________________

## Optional (Level up) 

### Hierarchical Agglomerative Clustering

Now that we've tried doing market segmentation with k-means clustering, let's end this lab by trying with HAC!

In the cells below, use [Agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) to make cluster predictions on the datasets we've created and see how HAC's performance compares to k-mean's performance. 

**_NOTE_**: Don't just try HAC on the PCA-transformed dataset -- also compare algorithm performance on the scaled and unscaled datasets, as well! 


```python
from sklearn.cluster import AgglomerativeClustering

```


```python
# Instantiate AgglomerativeClustering
hac_unscaled = AgglomerativeClustering(n_clusters=2)

# Fit the HAC model to the unscaled data and predict clusters
cluster_preds_hac_unscaled = hac_unscaled.fit_predict(df)

# Calculate the Adjusted Rand Index for the unscaled data
ari_score_hac_unscaled = adjusted_rand_score(channels, cluster_preds_hac_unscaled)

# Display the Adjusted Rand Index for the unscaled data
print(f"Adjusted Rand Index (Unscaled Data, HAC): {ari_score_hac_unscaled}")

```


```python
# Instantiate AgglomerativeClustering
hac_scaled = AgglomerativeClustering(n_clusters=2)

# Fit the HAC model to the scaled data and predict clusters
cluster_preds_hac_scaled = hac_scaled.fit_predict(scaled_df)

# Calculate the Adjusted Rand Index for the scaled data
ari_score_hac_scaled = adjusted_rand_score(channels, cluster_preds_hac_scaled)

# Display the Adjusted Rand Index for the scaled data
print(f"Adjusted Rand Index (Scaled Data, HAC): {ari_score_hac_scaled}")

```


```python
# Instantiate AgglomerativeClustering
hac_pca = AgglomerativeClustering(n_clusters=2)

# Fit the HAC model to the PCA-transformed data and predict clusters
cluster_preds_hac_pca = hac_pca.fit_predict(pca_data_reduced)

# Calculate the Adjusted Rand Index for the PCA-transformed data
ari_score_hac_pca = adjusted_rand_score(channels, cluster_preds_hac_pca)

# Display the Adjusted Rand Index for the PCA-transformed data
print(f"Adjusted Rand Index (PCA-Transformed Data, HAC): {ari_score_hac_pca}")

```
## Summary

In this lab, you used your knowledge of clustering to perform a market segmentation on a real-world dataset. You started with a cluster analysis with poor performance, and then implemented some changes to iteratively improve the performance of the clustering analysis!
