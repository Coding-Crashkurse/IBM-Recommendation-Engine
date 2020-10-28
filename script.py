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
df['article_id'] = df['article_id'].astype('int')

### Part I : Exploratory Data Analysis
df["email"].value_counts()
sns.displot(data=df["email"].value_counts())

count_per_user = df.groupby("email").count()["article_id"]
median_val = count_per_user.median()
max_views_per_user = count_per_user.max()

# 2 Explore and remove duplicates from the df_content dataframe
duplicates = df_content[df_content["article_id"].duplicated(keep=False)].sort_values(by=["article_id"])
duplicates

df_content.drop_duplicates(subset=["article_id"], inplace=True, keep="first")

#3 Use the cells below to find:
df.nunique()
df.shape[0]

df["article_id"].value_counts().head(10)
   
def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()

# Part I Rank based Recommendations
def get_top_articles(n, df=df):
    return df["title"].value_counts().head(n).reset_index()["index"].to_list()

def get_top_article_ids(n, df=df):
    return df["article_id"].value_counts().head(n).reset_index()["index"].to_list()

# Part III User-User based Collaborative Filtering
def create_user_item_matrix(df):
    user_item = df.groupby(["user_id", "article_id"])["title"].max().unstack()
    user_item = user_item.notnull() * 1
    return user_item

user_item = create_user_item_matrix(df)
user_item
### Find similar users


