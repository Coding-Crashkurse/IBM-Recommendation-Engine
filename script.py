import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

df = pd.read_csv(
    "C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/user-item-interactions.csv"
)
df_content = pd.read_csv(
    "C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/articles_community.csv"
)
del df["Unnamed: 0"]
del df_content["Unnamed: 0"]
df.head()
df["article_id"] = df["article_id"].astype("int")

### Part I : Exploratory Data Analysis
df["email"].value_counts()
sns.displot(data=df["email"].value_counts())

count_per_user = df.groupby("email").count()["article_id"]
median_val = count_per_user.median()
max_views_per_user = count_per_user.max()

# 2 Explore and remove duplicates from the df_content dataframe
duplicates = df_content[df_content["article_id"].duplicated(keep=False)].sort_values(
    by=["article_id"]
)
duplicates

df_content.drop_duplicates(subset=["article_id"], inplace=True, keep="first")

# 3 Use the cells below to find:
df.nunique()
df.shape[0]

df["article_id"].value_counts().head(10)


def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df["email"]:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


email_encoded = email_mapper()
del df["email"]
df["user_id"] = email_encoded

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
def find_similar_users(user_id, user_item=user_item):
    similarity = dict()
    for user in user_item.index:
        similarity.update({user: np.dot(user_item.loc[user_id], user_item.loc[user])})

    similarity.pop(user_id)
    sorted_similarity = sorted(similarity.items(), key=lambda kv: kv[1], reverse=True)
    most_similar_users = [_id[0] for _id in sorted_similarity]

    return most_similar_users  # return a list of the users in order from most to least similar


print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))


### 3.  Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.
# Complete the functions below to return the articles you would recommend to each user.
def get_article_names(article_ids, df=df):
    articles_df = df[df["article_id"].isin(article_ids)]
    articles_names = articles_df["title"].drop_duplicates().tolist()
    return articles_names


def get_user_articles(user_id, user_item=user_item):
    article_ids = user_item.loc[user_id][user_item.loc[user_id] == 1].index.to_list()
    article_names = get_article_names(article_ids)
    return article_ids, article_names  # return the ids and names


def user_user_recs(user_id, m=10):
    recs = []
    own_ids, own_articles =  get_user_articles(user_id)
    most_similar_users = find_similar_users(user_id)
    
    for user in most_similar_users:
        article_ids, article_names =  get_user_articles(user)
        unseen_articles_ids = [x for x in article_ids if x not in own_ids]
        recs.extend(unseen_articles_ids)
        if len(recs) >= m:
            break
    recs = recs[0:m]
    return recs

get_article_names(user_user_recs(4, m = 3))



