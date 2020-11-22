import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

df = pd.read_csv("C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/user-item-interactions.csv")
df_content = pd.read_csv("C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/data/articles_community.csv")
del df["Unnamed: 0"]
del df_content["Unnamed: 0"]
df.head()
#df["article_id"] = df["article_id"].astype("int")

### Part I : Exploratory Data Analysis
df["email"].value_counts().hist()
df["email"].hist()

count_per_user = df.groupby("email").count()["article_id"]
count_per_user
median_val = count_per_user.median()
max_views_per_user = count_per_user.max()

# 2 Explore and remove duplicates from the df_content dataframe
duplicates = df_content[df_content["article_id"].duplicated(keep=False)].sort_values(by=["article_id"])
duplicates

df_content.drop_duplicates(subset=["article_id"], inplace=True, keep="first")

# 3 Use the cells below to find:
df.nunique()
df.shape[0]
df["article_id"].value_counts().head(10)

df_content.nunique()



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

list(df["title"].value_counts().head(3).reset_index()["index"])

get_top_articles(5)

def get_top_article_ids(n, df=df):
    article_ids = list(df["article_id"].value_counts().head(n).reset_index()["index"])
    article_ids = [str(x) for x in article_ids]
    return 


get_top_article_ids(10)

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
    articles_names = list(articles_df["title"].drop_duplicates())
    return articles_names


def get_user_articles(user_id, user_item=user_item):
    article_ids = (user_item.loc[user_id][user_item.loc[user_id] == 1].index)
    article_ids = [str(x) for x in list(article_ids)]
    article_names = get_article_names(article_ids)
    return article_ids, article_names  # return the ids and names


def user_user_recs(user_id, m=10):
    recs = []
    own_ids, own_articles =  get_user_articles(user_id)
    most_similar_users = find_similar_users(user_id)
    
    for user in most_similar_users:
        article_ids, article_names =  get_user_articles(user)
        unseen_articles_ids = [str(x) for x in article_ids if x not in own_ids]
        recs.extend(unseen_articles_ids)
        if len(recs) >= m:
            break
    recs = recs[0:m]
    return recs

get_article_names(user_user_recs(1, m = 10))

### More consistent user_user_recs

def get_top_sorted_users(user_id, df=df, user_item=user_item):
    neighbor_ids = []
    similarity = []
    interactions = []
    
    for user in user_item.index:
        if user is not user_id:
            neighbor_ids.append(user)
            similarity.append(np.dot(user_item.loc[user_id], user_item.loc[user]))
            interactions.append(df[df['user_id']==user].shape[0])
            
    neighbors_df = pd.DataFrame({"neighbor_id": neighbor_ids, 'similarity': similarity, "num_interactions": interactions}).sort_values(by=["similarity", "num_interactions"], ascending=False)
    return neighbors_df


def user_user_recs_part2(user_id, m=10):
    recs = []
    own_ids, own_articles =  get_user_articles(user_id)
    
    top_sorted_users = get_top_sorted_users(user_id)
    
    for user in top_sorted_users.neighbor_id:
        article_ids, article_names =  get_user_articles(user)
        unseen_articles_ids = [str(x) for x in article_ids if x not in own_ids]
        recs.extend(unseen_articles_ids)
        if len(recs) >= m:
            break
    recs = recs[0:m]
    rec_names = get_article_names(recs)
    
    return recs, rec_names  


new_user_recs = get_top_article_ids(10)
new_user_recs

# Test your functions here - No need to change this code - just run this cell


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

user_item_matrix = pd.read_pickle('C:/Users/User/Desktop/Fortbildung/Projects/RecommendationEngine/user_item_matrix.p')

u, s, vt = np.linalg.svd(user_item_matrix)# use the built in to get the three matrices

num_latent_feats = np.arange(10,700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)
    
    
plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');

df_train = df.head(40000)
df_test = df.tail(5993)

# Your code here
user_item_train = create_user_item_matrix(df_train)
user_item_test = create_user_item_matrix(df_test)


def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    # Your code here
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    test_idx = list(user_item_test.index)
    test_arts = list(user_item_test.columns)
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)


u_train, s_train, vt_train = np.linalg.svd(user_item_train)

train_idx = user_item_train.index
train_arts = user_item_train.columns

common_idx =  user_item_train.index.isin(test_idx)
common_arts = user_item_train.columns.isin(test_arts)

u_test = u_train[common_idx]
vt_test = vt_train[:, common_arts]

predictable_df = user_item_test.loc[user_item_train[user_item_train.index.isin(test_idx)].index]

# initialize testing parameters
num_latent_feats = np.arange(10,700+10,20)
sum_errs_test = []

for k in num_latent_feats:
    # restructure with k latent features for both training and test sets
    s_train_new, u_train_new, vt_train_new = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
    u_test_new, vt_test_new = u_test[:, :k], vt_test[:k, :]
    
    user_item_test_est = np.around(np.dot(np.dot(u_test_new, s_train_new), vt_test_new))
    
    diffs = np.subtract(predictable_df , user_item_test_est)
    
    err_test = np.sum(np.sum(np.abs(diffs)))
    sum_errs_test.append(err_test)


plt.plot(num_latent_feats, 1 - np.array(sum_errs_test)/(predictable_df .shape[0] * predictable_df .shape[1]), label='Test')
plt.xlabel('# Latent Features')
plt.ylabel('Prediction Accuracy')
plt.legend()
plt.title('Test Accuracy vs. Number of Latent Features')
plt.show()




