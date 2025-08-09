#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

#Loading the dataset
data = pd.read_csv("dataset/marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))
print(data.head())

#Information on features 
print(data.info())


#To remove the NA values
data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))


data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format='mixed', dayfirst=True, errors='coerce')
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
#Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in therecords:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))

# Creating a feature ("Customer_For") of the number of days the customers started to shop in the store relative to the last recorded date

#Created a feature "Customer_For"
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

#Now we will be exploring the unique values in the categorical features to get a clear idea of the data.

print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())

#In the next bit, I will be performing the following steps to engineer some new features:
"""
    Extract the "Age" of a customer by the "Year_Birth" indicating the birth year of the respective person.
    Create another feature "Spent" indicating the total amount spent by the customer in various categories over the span of two years.
    Create another feature "Living_With" out of "Marital_Status" to extract the living situation of couples.
    Create a feature "Children" to indicate total children in a household that is, kids and teenagers.
    To get further clarity of household, Creating feature indicating "Family_Size"
    Create a feature "Is_Parent" to indicate parenthood status
    Lastly, I will create three categories in the "Education" by simplifying its value counts.
    Dropping some of the redundant features
"""

#Feature Engineering
#Age of customer today 
data["Age"] = 2025-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

#Segmenting education levels in three groups
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

#Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)

#Now that we have some new features let's have a look at the data's stats.
print(data.describe())

#Do note that max-age is 132 years, As I calculated the age that would be today (i.e. 2025) and the data is old.

#Plotting following features
To_Plot = [ "Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
plt.figure()
sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()
plt.savefig("Visuals/Plot Of Some Selected Features: A Data Subset")


#Dropping the outliers by setting a cap on Age and income. 
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(data))

#Next, let us look at the correlation amongst the features. (Excluding the categorical attributes at this point)
#correlation matrix 
corrmat = data.select_dtypes(include='number').corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True, center=0)
plt.show()
plt.savefig("Visuals/correlation amongst the features")
plt.close()

'''
    The following steps are applied to preprocess the data:
    Label encoding the categorical features
    Scaling the features using the standard scaler
    Creating a subset dataframe for dimensionality reduction

'''
#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")

#Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")

#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
print(scaled_ds.head())

# Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables.

#Steps in this section:
'''
    Dimensionality reduction with PCA
    Plotting the reduced dataframe
'''
#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
print(PCA_ds.describe().T)

# visualizing pca 

x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()
plt.savefig("Visuals/A 3D Projection Of Data In The Reduced Dimension")
plt.close()

'''
    Now that I have reduced the attributes to three dimensions, 
    I will be performing clustering via Agglomerative clustering. 
    Agglomerative clustering is a hierarchical clustering method. 
    It involves merging examples until the desired number of clusters is achieved.
'''

# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()
plt.savefig("Visuals/Elbow method")


#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
data["Clusters"]= yhat_AC

cmap = []
#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o')
ax.set_title("The Plot Of The Clusters")
plt.show()
plt.savefig("Visuals/agglomerative clustering")

#EVALUATING MODELS

#Plotting countplot of clusters
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=data["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
plt.show()
plt.savefig("Visuals/Distribution Of The Clusters ")




pl = sns.scatterplot(data = data,x=data["Spent"], y=data["Income"],hue=data["Clusters"], palette= pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()
plt.savefig("Visuals/Cluster's Profile Based On Income And Spending")


#From the above plot, it can be clearly seen that cluster 1 is our biggest set of customers closely followed by cluster 0. 
# We can explore what each cluster is spending on for the targeted marketing strategies.


#Creating a feature to get a sum of accepted promotions 
data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Total_Promos"],hue=data["Clusters"], palette= pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()
plt.savefig("Visuals/Count Of Promotion Accepted")

"""
    There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one part take in all 5 of them.
      Perhaps better-targeted and well-planned campaigns are required to boost sales.
"""

#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=data["NumDealsPurchases"],x=data["Clusters"], palette= pal)
pl.set_title("Number of Deals Purchased")
plt.show()
plt.savefig("Visuals/Number of Deals Purchased")

"""
    Unlike campaigns, the deals offered did well. It has best outcome with cluster 0 and cluster 3. However, 
    our star customers cluster 1 are not much into the deals. Nothing seems to attract cluster 2 overwhelmingly
"""


# profiling the clusters formed and come to a conclusion about 
# who is our star customer and who needs more attention from the retail store's marketing team.

Personal = [ "Kidhome","Teenhome","Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Spent"], hue =data["Clusters"], kind="kde", palette=pal)
    plt.show()
    
    plt.savefig(f"Visuals/Plot for {i} Vs Spent")

