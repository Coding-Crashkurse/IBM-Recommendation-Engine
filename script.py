import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

df = pd.read_csv("C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/user-item-interactions.csv")
df_content = pd.read_csv("C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/articles_community.csv")
del df['Unnamed: 0']
del df_content['Unnamed: 0']
df.head()

### Part I : Exploratory Data Analysis
df["email"].value_counts()
sns.displot(data=df["email"].value_counts())

count_per_user = df.groupby("email").count()["article_id"]
median_val = count_per_user.median()
max_views_per_user = count_per_user.max()

# 2 Explore and remove duplicates from the df_content dataframe
duplicates = df_content[df_content["article_id"].duplicated(keep=False)].sort_values(by=["article_id"])
duplicates

df_content.drop_duplicates(subset=["article_id"], inplace=True)
