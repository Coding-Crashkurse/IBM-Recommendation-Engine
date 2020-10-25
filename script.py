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
