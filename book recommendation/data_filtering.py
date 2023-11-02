import numpy as np
import pandas as pd 
import pickle
from sklearn.metrics.pairwise import cosine_similarity

books=pd.read_csv('data/books.csv')
user=pd.read_csv('data/users.csv')
ratings=pd.read_csv('data/ratings.csv')

# print(books.shape)

# popularity based recommender s/y
ratings_with_name = ratings.merge(books,on='ISBN')

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
# print(num_rating_df.head())

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')

# taking only those books whose num of ratings is more than 250 and taking only top 50 
popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)
popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# print(popular_df)


# now lets work on the collaborative filtering based recommendation s/y

x = ratings_with_name.groupby('User-ID').count()['Book-Title'] > 200

padhe_likhe_users = x[x].index

# getting ratings only from the experienced reviewers
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# getting only the books which have reviews more than 50
y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>= 50
famous_books = y[y].index

# getting the ratings of only the famours books
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)

# print(pt)

# lets find the similarities b/w the books
similarity_scores = cosine_similarity(pt)

# -----------------------------------------------------------

# now lets start writing the functions for book recommendations

def recommend(book_name):
    index=np.where(pt.index == book_name)[0][0]

    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]

    data=[]

    for i in similar_items:
        item=[]
        temp_df = books[books['Book-Title']==pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data

# print(recommend('Animal Farm'))

books.drop_duplicates('Book-Title')

# export popular books df
pickle.dump(popular_df,open('data/popular_books.pkl','wb'))

# export pt table, books, similarity_score
pickle.dump(pt,open('data/pt.pkl','wb'))
pickle.dump(books,open('data/books_nd','wb'))
pickle.dump(similarity_scores,open('data/similarity_scores','wb'))

