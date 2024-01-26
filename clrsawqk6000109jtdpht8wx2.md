---
title: "K-means Cluster Model"
seoTitle: "clustering"
datePublished: Wed Jan 24 2024 21:32:25 GMT+0000 (Coordinated Universal Time)
cuid: clrsawqk6000109jtdpht8wx2
slug: k-means-cluster-model
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1706050314167/0ff818da-5a88-40fd-a13b-04ba66c57b76.png
tags: machine-learning, cluster, k-means-clustering, machine-learning-models

---

## Introduction

Identifying patterns and structure data is a key to success in Data Science. There are many ways out there to accomplish this task. However, one way to make it fast and using Machine Learning is with Clustering models.

Here are some of the benefits of using these models:

* Reduce data dimensionality
    
* Detect anomalies and outliers
    
* Create new features for further ML models
    
* Explore complex data and unlabeled datasets
    

Clustering in machine learning is a technique that groups unlabeled data points into different clusters based on some similarity measures.

## K-Means

There are many libraries available for python that provides clustering tools but we will use K-Means algorithm on this article. K-Means is part of scikit-learn library, it is easy to use and as scikit-learn provides more Machine Learning and Data Analytics tools its a good option. It is always convenient to have all tools in one place.

## The Dataset

The **Amazon Data Science Books Dataset** is stored in [Kaggle](https://www.kaggle.com/datasets/die9origephit/amazon-data-science-books). You can use Kaggle for free and you can sign in with a google account.

Per the description on Kaggle "*The dataset contains 946 books obtained from scraping Amazon books related to data science, statistics, data analysis, Python, deep learning, and machine learning.*"

The objective of the presented code is to label the data based on all features given using the K-Means algorithm. We will also create a `weighted_rating` feature to compare the cluster labels to determine if the clustering model did a good job labeling the data. In real world projects this is not necessary. The intent is also to prepare unlabeled data for future analysis (EDA, ML, DP, etc.).

Here are the steps of the project:

1. Data Collection
    
2. Data Cleaning
    
3. Feature Engineering
    
4. Cluster Model
    
5. Next Steps
    

## Data Collection

We first import all needed libraries and tools.

```python
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
```

`pandas` and `numpy` will help to make most of the cleaning in our data, `matplotlib` and `seaborn` will be use to visualize the data and `sklearn` will provide the K-means model, `MinMaxScaler` to scale our output data.

Next, we collect the data from the saved csv file. ***Make sure you saved the file from Kaggle before running the code.***

```python
df = pd.read_csv('final_book_dataset_kaggle2.csv')
df.head()
```

We can review the first five rows of the data to make a visual exploration

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706051315885/843594bb-ee49-42b7-b061-3c97386bb335.png align="center")

## Data Cleaning

Once the data is loaded in a Pandas DataFrame, lets count the null values and plot them in a bar chart.

```python
null_counts = df.isna().sum()

# Visualize the nulls
null_counts.plot(kind='barh')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706051481226/0998babb-d89e-4d7e-877d-e365436e3c35.png align="center")

The process of cleaning data includes:

* Replacing `null` values
    
* Removing extra characters
    
* Replacing data types
    

But even though we know what columns have null values it is important to indicate what values will replace the nulls. For that we can use python lists to hold the columns names to identify the type of null replacement.

```python
# Columns to replace with 0:
stars_cols = list(df.loc[:, df.columns.str.startswith('star')].columns)
null_to_zero = ['dimensions','weight', 'n_reviews', 'avg_reviews', 'pages', 'price', 'price (including used books)'] + stars_cols

# Columns to replace wiht 'No Provided'
null_to_np = ['ISBN_13', 'publisher', 'language', 'author']
null_to_np

# Replace as needed
df[null_to_zero] = df[null_to_zero].fillna(0)
df[null_to_np] = df[null_to_np].fillna('No Provided')
```

`starts_cols` are stored separately just because it is convenient for latter transformations.

`null_to_zero` are numerical columns, therefore will be replaced with 0.

`null_to_np` are columns that contains text and nulls be replaced with **'No Provided'**.

Finally we replace as defined.

Next step is replacing the data types. There are some `object` columns that need to be numerical columns. But also, some of those columns have extra characters no needed. We fix that with the following code.

```python
# First we remove extra chars
df[stars_cols] = df[stars_cols].replace('%', '', regex=True)
df['dimensions'] = df['dimensions'].replace(' inches', '', regex=True)
df['author'] = df['author'].replace([r'\[',r'\]',], ['',''], regex=True)
df['weight'] = df['weight'].replace([' pounds', ' ounces'], ['',''], regex=True)

# Remove all none numerical values and replace with 0
df['weight'] = pd.to_numeric(df['weight'], errors ='coerce').fillna(0)
df['pages'] = pd.to_numeric(df['pages'], errors ='coerce').fillna(0)

#Remove thousands comma
df[['pages','n_reviews']] = df[['pages','n_reviews']].replace(',', '', regex=True)

# Keep just one value, first value
df['price (including used books)'] = df['price (including used books)'].str.split(' -').str[0]
```

Once we got rid of chars, data types are replaced as needed.

```python
# Change Dtypes
df[stars_cols] = df[stars_cols].astype(int) / 100 #To decimals
df[['weight', 'price (including used books)', 'price', 'avg_reviews']] = df[['weight', 'price (including used books)', 'price', 'avg_reviews']].astype(float)
df[['pages', 'n_reviews']] = df[['pages', 'n_reviews']].astype(int)
df.head()
```

Our data is now cleaned and in the right format !!

## Feature Engineering

Let's get some extra meaningful columns from our dataset.

The idea is extract details from existing columns that will improve any Machine Learning model. Here is a summary of final columns:

* Extract dimension values - `dimensions` column contains the cover width, flap width and the cover height. These data points are more useful individually.
    
* Weighted Rating - Our control metric. Remember that this will be compared to the clusters generated by the model.
    

The following code will split the values from the `dimensions` column by the `' x '` delimiter and create three new columns.

Then drop the `dimensions` column.

```python
 # cover_width X flap_width X cover_height
df[['cover_width','flap_width','cover_height']] = df['dimensions'].str.split(' x ', expand=True)
# Replace dtype
df['cover_width'] = pd.to_numeric(df['cover_width'], errors ='coerce').fillna(0)
df['flap_width'] = pd.to_numeric(df['flap_width'], errors ='coerce').fillna(0)
df['cover_height'] = pd.to_numeric(df['cover_height'], errors ='coerce').fillna(0)
df[['cover_width','flap_width','cover_height']] = df[['cover_width','flap_width','cover_height']].astype(float)
# Drop dimensions column
df.drop('dimensions', axis=1, inplace=True)

df[['cover_width','flap_width','cover_height']].head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706127073307/d1a5ba3f-10af-4a00-909a-f91b6468fd32.png align="center")

Now we create the control column, `weighted_rating`.

```python
# create the weighted_rating by multiplying every star column by its count of stars, then sum all, then devide by 100
df['weighted_rating'] = (df['star1']*1 + df['star2']*2 + df['star3']*3 + df['star4']*4 + df['star5']*5) / 100
```

We are now ready to create our cluster model !!

## Cluster Model

### Numerical values only

Before fitting data to any Machine Learning model it is required to make sure there are numerical values only. On this case there are some `object` columns in the data frame. Lets fix that by selecting the categorical columns then get codes for each category. We also want to keep the original columns for later analysis so the new coded columns should have an identifier.

```python
# Make category codes, keep original columns
# Save the name of the new columns to acces later

# Get categorical columns in a list
c = (df.dtypes == 'object')
categorical_cols = list(c[c].index)
categorical_codes = []

# Iterate each column, conver to category,
# assign a code 
# then save the name of the column
for col in df[categorical_cols]:
  col_name = col + '_cat'
  categorical_codes.append(col_name)
  df[col + '_cat'] = df[col].astype('category').cat.codes

categorical_codes
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706127450445/cbc2344b-7a06-41e4-94a6-188ec1f337b1.png align="center")

### Elbow Method

The cluster model identifies the patterns on the data and assigns a **label** based on its findings. However, it **does not** assign the number of clusters *(buckets, labels or categories of the data)*. To help the model on this task we can use the **Elbow Method**.

The elbow method is a technique to find the optimal number of clusters in a data set using the K-means algorithm. It involves plotting the within-cluster sum of squares (**WCSS**) against the number of clusters and looking for the point where the curve bends or levels off.

```python
# Check inertias, elbow method
inertias = []

# Get numerical and categorical code columns
numerical_cols = df.select_dtypes(include=['int', 'float']).columns
cols = categorical_codes + list(numerical_cols)

# Remove weighted_rating, as it is our control variable
# This will prevent from bias the clustering selection
cols.remove('weighted_rating')

df[cols] = df[cols].fillna(0)
for i in range(1,11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(df[cols])
  inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706127861262/f58c1cfd-29b6-4581-8b01-f3e1a449a225.png align="center")

This point indicates that adding more clusters does not significantly improve the model performance and hence it is the best choice for the number of clusters. 3 is the number of clusters for our data.

### K-Means model

By calling the `KMeans()` class we create a K-means model, we pass the number of clusters defined by the Elbow Method `KMeans(n_clusters=3)` . The result is a numerical label list in this case from 0 to 2. We also replace these values with text for better visualization.

```python
# Define the model
kmeans = KMeans(n_clusters=3)
# Fit the model
kmeans.fit(df[cols])
# Get the labels from fitted model, then replace them with text
df['Cluster'] = kmeans.labels_
df['Cluster'].replace({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}, inplace=True)

df.head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706128327625/4a34b291-7ee3-423d-b8b3-9a5732232bce.png align="center")

Lets plot some data. We can use our control column `weighted_rating` and `n_reviews` to see how the model labeled this data.

```python
plt.figure(figsize=(15, 5))
# Plot data with cluster
sns.scatterplot(data=df,
                x='n_reviews', y='weighted_rating',
                c=kmeans.labels_)
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706129251173/68bd4779-d029-434b-864c-6ee6930ddd25.png align="center")

As we can see the cluster model did a good job at identifying these three groups of data. But we can review in more detail the predicted labels and compare to our control column `weighted_rating` .

We can simply visualize the count of movie titles by cluster or review the raw data of this count.

```python
# Visualize Count of records y clusters
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.countplot(df['Cluster'])
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706129440675/b3493453-4e82-41f4-a2ee-53877f64deb7.png align="center")

```python
# See the raw numbers
clusters_count = df.groupby('Cluster')['title'].count()

clusters_count
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706129469114/437f0661-9783-4bdf-9b89-2c87240a424a.png align="center")

With the above outputs we can have an idea of proportions on each cluster. By looking at it we can tell:

* The amount of titles that belongs to cluster 1 is higher that the other two clusters.
    
* The cluster 2 contains just one record and may be an outlier.
    

But, how does it compare to the control column `weighted_rating` ?

```python
# weighted_rating by cluster and cluster count
weighted_rating_cluster = df.groupby('Cluster').agg({'weighted_rating':sum, 'title': np.size})
weighted_rating_cluster
```

By running above code we can compare both measures, count of `titles` and the sum of `weighted_rating` grouped by the clusters.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706129719371/43ffe2fe-1e3f-401e-b52b-918d225d7f11.png align="center")

Just by looking at it we can land on the same conclusions

* The sum of `weighted_rating` that belongs to cluster 1 is higher that the other two clusters.
    
* The cluster 2 is too low and may be an outlier.
    

We can reveal this comparative conclusions even further by scaling our data. I will not explain this technique with much details but I do recommend searching more online. Basically the two columns values are way different, the control column contains tens and decimals. The count of titles is on units, tens and hundreds.

The `MinMaxScaler()` class from `sklearn` can transform this data to a scale of 0 to 1. after the transformation we can now compare apples to apples !!

```python
# Rescale the columns to compare
scaler = MinMaxScaler()
weighted_rating_cluster_scaled = pd.DataFrame(scaler.fit_transform(weighted_rating_cluster), columns=weighted_rating_cluster.columns)
weighted_rating_cluster_scaled
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706130193283/c75bacc6-2cef-4a80-96bb-ea8f3dcbb93a.png align="center")

Both columns are really similar. Lets compare in a graph.

```python
# Plot Scaled df
weighted_rating_cluster_scaled.plot(kind='bar')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1706130265734/b3209e98-b866-45e1-b046-846bcfa7629d.png align="center")

We can conclude that the cluster model did a good job on detecting the patterns of our data !!

### Next Steps

This is just one simple example of using K-means models and I do recommend investigating more about cluster model in Machine Learning. Now that the data is labeled there are many possibilities to use this data for, here some ideas:

* Correlation Analysis - Find correlations between all columns and the cluster label. This will reveal the patters found in the K-mean model. Also you can discover more features to be implemented.
    
* Machine Learning Models - Data is now labeled, why to stop here?. You can create a classification model on top of the cluster model to make predictions of how new data may be clustered.
    

You can access to my [GitHub repo](https://github.com/abrahamfullstack/Machine_Learning/blob/main/CLUSTERS_Amazon_Data_Science_Books_Dataset.ipynb) and download the full code and modify as you desire.

I hope you enjoy this article. Keep on learning.