# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("dataset/marketing_campaign.csv", sep="\t")
    data = data.dropna()
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], errors='coerce')
    data["Customer_For"] = (data["Dt_Customer"].max() - data["Dt_Customer"]).dt.days
    data["Age"] = 2025 - data["Year_Birth"]
    data["Spent"] = data[["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]].sum(axis=1)
    data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
    data["Children"] = data["Kidhome"] + data["Teenhome"]
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2}) + data["Children"]
    data["Is_Parent"] = np.where(data["Children"]> 0, 1, 0)
    data["Education"] = data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
    data = data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)
    data = data[(data["Age"]<90) & (data["Income"]<600000)]

    # Label Encoding
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    for i in object_cols:
        data[i] = LabelEncoder().fit_transform(data[i])

    return data

data = load_data()

# Sidebar
st.sidebar.title("Customer Segmentation")
st.sidebar.write("Using Agglomerative Clustering + PCA")

st.sidebar.header("🔍 Predict Cluster for New Customer")

with st.sidebar.form("customer_form"):
    income = st.number_input("Income", min_value=0, max_value=300000, step=5000)
    recency = st.slider("Recency (days since last purchase)", 0, 100, 10)
    customer_for = st.slider("Customer For (days since enrolled)", 0, 3650, 1000)
    age = st.slider("Age", 18, 100, 40)
    children = st.slider("Children (kids + teens)", 0, 5, 0)
    is_parent = st.selectbox("Is Parent?", ["No", "Yes"])
    education = st.selectbox("Education", ["Undergraduate", "Graduate", "Postgraduate"])
    living_with = st.selectbox("Living With", ["Alone", "Partner"])
    spent = st.number_input("Total Amount Spent", min_value=0, max_value=10000, step=100)

    submitted = st.form_submit_button("Predict Cluster")
# PCA and clustering
features = data.drop(columns=['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Complain','Response'])
# Drop NaNs in both features and data
valid_index = features.dropna().index
features = features.loc[valid_index]
data = data.loc[valid_index]

# Proceed with scaling + PCA
scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled)
ac = AgglomerativeClustering(n_clusters=4)
clusters = ac.fit_predict(pca_data)



data["Cluster"] = clusters

if submitted:
    # Encode categorical fields
    is_parent_val = 1 if is_parent == "Yes" else 0
    education_val = {"Undergraduate": 0, "Graduate": 1, "Postgraduate": 2}[education]
    living_val = {"Alone": 0, "Partner": 1}[living_with]
    family_size = (1 if living_with == "Alone" else 2) + children

    # Create new record
    user_row = {
        'Income': income,
        'Recency': recency,
        'Customer_For': customer_for,
        'Age': age,
        'Children': children,
        'Family_Size': family_size,
        'Is_Parent': is_parent_val,
        'Education': education_val,
        'Living_With': living_val,
        'Spent': spent
    }

    # Use same columns as features used earlier
    input_df = pd.DataFrame([user_row])
    combined = pd.concat([features, input_df], ignore_index=True)
    combined = combined.dropna()

    # Scale + PCA
    scaled_combined = StandardScaler().fit_transform(combined)
    pca_combined = PCA(n_components=3).fit_transform(scaled_combined)

    # Run clustering again
    predicted_clusters = AgglomerativeClustering(n_clusters=4).fit_predict(pca_combined)
    predicted_label = predicted_clusters[-1]

    st.sidebar.success(f"Predicted Cluster for User: 🎯 **Cluster {predicted_label}**")
    cluster_messages = {
    0: "🧠 High-value cluster: Loyal customers!",
    1: "💼 Moderate income & engagement. Upsell possible.",
    2: "🔍 Low spenders, high recency. Need attention.",
    3: "🎯 Deal-responsive group. Good for promos."
    }
    # Optional: Show user PCA location
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_combined[:-1, 0], pca_combined[:-1, 1], pca_combined[:-1, 2], c=predicted_clusters[:-1], cmap='tab10', label='Existing')
    ax.scatter(pca_combined[-1, 0], pca_combined[-1, 1], pca_combined[-1, 2], c='black', s=100, label='You')
    ax.set_title("PCA + Cluster Visualization")
    ax.legend()
    st.sidebar.pyplot(fig)

st.sidebar.info(cluster_messages.get(predicted_label, "Unknown cluster"))
    
st.subheader("Cluster-wise Spending vs Income")
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x="Income", y="Spent", hue="Cluster", palette="tab10", ax=ax)
st.pyplot(fig)

st.subheader("Cluster Distribution")
st.bar_chart(data["Cluster"].value_counts())

st.subheader("View Clustered Data")
st.dataframe(data)




