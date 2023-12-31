from flask import Flask, render_template, request
import pickle
import numpy as np


# lets first import the data files with working data
popular_df = pickle.load(open('data/popular_books.pkl','rb'))
pt = pickle.load(open('data/pt.pkl','rb'))
books = pickle.load(open('data/books_nd','rb'))
similarity_scores = pickle.load(open('data/similarity_scores','rb'))

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values))


app.route('/recommender')
def recommend_ui():
    return render_template('recommender.html')

app.route('/recommender', method=['post'])
def recommender():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)


    return render_template('recommender.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)