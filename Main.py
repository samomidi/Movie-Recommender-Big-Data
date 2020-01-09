import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# df = pd.read_csv('C:/Users/so185059/Documents/Python/Movie-rating/ml-20m/ratings.csv', sep='\t',
#                  names= ['user_id','item_id','rating','titmestamp'],
#                  dtype= {'userID':int, 'movieId':int, 'rating':int, 'timestamp':int})
# tp = pd.read_csv('C:/Users/so185059/Documents/Python/Movie-rating/ml-20m/ratings.csv',
#                  iterator=True, chunksize=200000, low_memory = False)
# df = pd.concat(tp, ignore_index=True)

mylist = []
for chunk in  pd.read_csv('C:/Users/so185059/Documents/Python/Movie-rating/ml-20m/ratings.csv', delimiter=',', chunksize=20000):
    mylist.append(chunk)
ratings = pd.concat(mylist, axis= 0)
del mylist

# print(ratings)


mylist = []
for chunk in  pd.read_csv('C:/Users/so185059/Documents/Python/Movie-rating/ml-20m/movies.csv', delimiter=',', chunksize=20000):
    mylist.append(chunk)
titles = pd.concat(mylist, axis= 0)
del mylist

# print(titles)



# df = pd.merge(ratings, titles, left_on= 'movieId', right_on='movieId')
# df = pd.merge(ratings,titles, on='mid', how='left')

intersect = ratings[['userId', 'movieId', 'rating', 'timestamp']].merge(titles[['movieId', 'title', 'genres']],
                                                                        on='movieId', how='left')
# intersect.to_csv('merged.csv', sep=',')


print(intersect.head())
print(intersect.describe())


average = pd.DataFrame(intersect.groupby('title')['rating'].mean())

average['number_of_ratings'] = intersect.groupby('title')['rating'].count()

average.head()

import matplotlib.pyplot as plt

average['rating'].hist(bins=50)

average['number_of_ratings'].hist(bins=60)


import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=average)


movie_matrix = intersect.pivot_table(index='userId', columns='title', values='rating')

# print(movie_matrix.head())

average.sort_values('number_of_ratings', ascending=False).head()


AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']

similar_to_air_force_one = movie_matrix.corrwith(AFO_user_rating)
similar_to_contact = movie_matrix.corrwith(contact_user_rating)

# add a header to the only column
corr_contact = pd.DataFrame(similar_to_contact, columns=['correlation'])
corr_contact.dropna(inplace=True)
corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace=True)


# add number of ratings column
corr_AFO = corr_AFO.join(average['number_of_ratings'])
corr_contact = corr_contact.join(average['number_of_ratings'])

# setting threshold
print(corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False))
print(corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False))


