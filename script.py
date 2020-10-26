import numpy as np
import pandas as pd

interactions = pd.read_csv("C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/user-item-interactions.csv")
articles = pd.read_csv("C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/articles_community.csv")

counts_per_user = interactions.groupby(["email"]).count()["article_id"]
counts_per_user.hist()

counts_per_user.mean()
counts_per_user.median()

# Find and explore dupicate articles
duplicates = articles[articles['article_id'].duplicated(keep=False)]

unique_articles = articles.drop_duplicates(subset=["article_id"], keep="first")

# Number of unique articles that have interaction with a user
interactions["article_id"].nunique()
# The of unique articles in the dataset
articles["article_id"].nunique()
# Number of unique users in the dataset
interactions["email"].nunique()
# Number of user-article interactions in the dataset
interactions.shape[0]

## Most viewed article id
interactions.groupby(["article_id"], as_index=False).count()
   
def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in interactions['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del interactions['email']
interactions['user_id'] = email_encoded

# show header
interactions.head()
articles.head()


### Get top articles
def get_top_articles(n, df):
    most_relevant_sorted = df["article_id"].value_counts().iloc[0:n]
    most_relevant_articles_df = pd.merge(most_relevant_sorted, articles, on="article_id", how="left")["doc_full_name"]
    return most_relevant_articles_df.to_list()

def get_top_article_ids(n, df):
    return df["article_id"].value_counts().iloc[0:n]


get_top_articles(10, interactions)
get_top_article_ids(10, interactions)


### Part 3 User-User Based Collaborative Filtering

user_item_matrix = interactions.groupby(["user_id", "article_id"]).agg(lambda x: 1).unstack().fillna(0).droplevel(0, axis = 1)

user_item_matrix.iloc[1] * user_item_matrix.iloc[1]


similarity = {}
for user in user_item_matrix.index:
    similarity[user] = np.dot(user_item_matrix.loc[1, :], user_item_matrix.loc[user, :])
    #similarity[user] = np.dot(user_item_matrix.iloc[1], user_item_matrix.iloc[user])
    
{k: v for k, v in sorted(similarity.items(), key=lambda item: item[1])}





