# IBM-Recommendation-Engine

This project is the submission for the Udacity Recommender System Project. The data is from the IBM Watson Studio platform and is used to make recommendations about new articles.

The project is devided in three parts:

<br>

## Part I: Exploratory Data Analysis

<br>
Get an idea of the two datasets by descriptive statistics and basic visuals.

<br>

## Part II: Rank-Based Recommendations

<br>

Provides recommendations based on historical popularity of the articles. This is important for new users.

<br>

## Part III: User-User Based Collaborative Filtering

<br>

For recommendations for existing users, the historical preferences of the user are important. To make better recommendations than just providing the most popular users, we can create and user-article-matrix and compute the similary of one user to all other users. The seen articles of the simliar users, which have not been seen by the user, will be recommended then.

If you want to run this project on you own computer, you have to install some libraries first:

<br>

Please run `pip install -r requirements.txt`
