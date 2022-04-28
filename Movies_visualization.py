#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler


#LOAD DATA
#df = pd.read_csv('D:/Project/movies.csv',index_col='movie_title')
df = pd.read_csv('movies.csv',index_col='movie_title')
df4 = df.copy()
df=df.sort_values('budget',ascending=False)
df2=df.head(10)
df3=df.head(20)
#df['imdb_score']=df['imdb_score'].round(decimals = 1)


#REGRESSION MODEL
#Clean Data
df4 = df4.dropna(axis=0)
df4_quant = df4.drop(['color',
                    'director_name',
                    'actor_1_name',
                    'actor_2_name',
                    'actor_3_name',
                    'genres',
                    'plot_keywords',
                    'movie_imdb_link',
                    'language',
                    'country',
                    'content_rating',
                    'aspect_ratio',
                    'title_year'],
                   axis=1)

#Split data into train and test sets
X = df4_quant.drop(columns='gross')
y = df4_quant['gross']
X_trn, X_vld, y_trn, y_vld = train_test_split(X, y, test_size=0.2, random_state=216)

#Standardize the data
scaler = StandardScaler()
X_trn_scaled = scaler.fit_transform(X_trn)
X_vld_scaled = scaler.transform(X_vld)

#Create and Train Lasso Model
alpha = 1
lasso_model = Lasso(alpha=alpha, max_iter=10000)
lasso_model.fit(X_trn_scaled, y_trn)


#STREAMLIT LAYOUT
#Config
st.set_page_config(layout='wide')

#Title
st.title('IMDB Movie Ratings')

#Select a Visualization Page
vis = st.sidebar.selectbox(label='Visualization Page', options=('Scores', 'Actors and Genres', 'Budgets', 'Prediction'))

#Scores
if vis == 'Scores':
    st.header('Movie Information')
    abc=pd.read_csv('movies.csv')
    title = st.text_input('Enter Movie title','Avatar')
    abc = abc[abc['movie_title'].str.contains(title)]
    st.write('About movie:', title)
    st.write(abc[['movie_title','actor_1_name','director_name','imdb_score','movie_imdb_link']])
    
    st.header('Top movies Under Selected Duration')
    option = st.selectbox('Select Duration',("<1.5 hr","1.5-2 hrs","2-2.5 hrs","2.5-3 hrs"))
    xyz=pd.read_csv('movies.csv')
    xyz=xyz.drop_duplicates(subset=['movie_title'])
    xyz=xyz.sort_values('imdb_score',ascending=False)
    xyz1 = xyz[xyz['duration'] <=90]
    xyz2 = xyz[xyz['duration'].between(90,120)]
    xyz3 = xyz[xyz['duration'].between(120,150)]
    xyz4 = xyz[xyz['duration'].between(150,180)]
    
    if(option=="<1.5 hr"):
        st.write('Top rated movies with time duration '+option)
        st.write(xyz1[['movie_title','duration', 'imdb_score']].head(5))
    if(option=="1.5-2 hrs"):
        st.write('Top rated movies with time duration '+option)
        st.write(xyz2[['movie_title','duration', 'imdb_score']].head(5))
    if(option=="2-2.5 hrs"):
        st.write('Top rated movies with time duration '+option)
        st.write(xyz3[['movie_title','duration', 'imdb_score']].head(5))
    if(option=="2.5-3 hrs"):
        st.write('Top rated movies with time duration '+option)
        st.write(xyz4[['movie_title','duration', 'imdb_score']].head(5))
    
    st.header('Distribution of IMDB Scores')
    plt.title("Ratings Count")
    fig = plt.figure(figsize=(30, 15))
    sns.countplot(x='imdb_score',data=df)
    st.pyplot(fig)
    
#Genres
elif vis == 'Actors and Genres':
    st.header('Top Rated Movies in Your Preferred Genre')
    option = st.selectbox('Select a Genre',("Action","Comedy","Adventure","Fantasy","Drama"))
    
    df_copy=pd.read_csv('movies.csv')
    if(option=="Action"):
        df_copy = df_copy[df_copy['genres'].str.contains(option)]
        st.write('Top rated movies in '+option+' genre:')
        st.write(df_copy[['movie_title']].head())

    if(option=="Comedy"):
        df_copy = df_copy[df_copy['genres'].str.contains(option)]
        st.write('Top rated movies in '+option+' genre:')
        st.write(df_copy[['movie_title']].head())

    if(option=="Adventure"):
        df_copy = df_copy[df_copy['genres'].str.contains(option)]
        st.write('Top rated movies in '+option+' genre:')
        st.write(df_copy[['movie_title']].head())

    if(option=="Fantasy"):
        df_copy = df_copy[df_copy['genres'].str.contains(option)]
        st.write('Top rated movies in '+option+' genre:')
        st.write(df_copy[['movie_title']].head())

    if(option=="Drama"):
        df_copy = df_copy[df_copy['genres'].str.contains(option)]
        st.write('Top rated movies in '+option+' genre:')
        st.write(df_copy[['movie_title']].head())
    
    st.header('Movies Based On Actors')
    option1 = st.selectbox('Select an Actor',("Johnny Depp","Chris Hemsworth","Robert Downey Jr","Christian Bale","Emma Stone"))
    df_copy1=pd.read_csv('movies.csv')
    df_copy1 = df_copy1.dropna()

    if(option1=="Johnny Depp"):
        df_copy1 = df_copy1[df_copy1['actor_1_name'].str.contains(option1)]
        st.write('List of '+option1+'\'s movies:')
        st.write(df_copy1[['movie_title']].head())

    if(option1=="Chris Hemsworth"):
        df_copy1 = df_copy1[df_copy1['actor_1_name'].str.contains(option1)]
        st.write('List of '+option1+'\'s movies:')
        st.write(df_copy1[['movie_title']].head())

    if(option1=="Robert Downey Jr"):
        df_copy1 = df_copy1[df_copy1['actor_1_name'].str.contains(option1)]
        st.write('List of '+option1+'\'s movies:')
        st.write(df_copy1[['movie_title']].head())

    if(option1=="Christian Bale"):
        df_copy1 = df_copy1[df_copy1['actor_1_name'].str.contains(option1)]
        st.write('List of '+option1+'\'s movies:')
        st.write(df_copy1[['movie_title']].head())

    if(option1=="Emma Stone"):
        df_copy1 = df_copy1[df_copy1['actor_1_name'].str.contains(option1)]
        st.write('List of '+option1+'\'s movies:')
        st.write(df_copy1[['movie_title']].head())
    
#Budget
elif vis == 'Budgets':    
    if st.sidebar.checkbox('Top 10 High Budget Movies'):
        st.write(df['budget'].head(10))
    if st.sidebar.checkbox('Top 10 Budget vs Number of Critics Reviews'):
        plt.title("Critics reviews")
        fig2 = plt.figure(figsize=(45, 20))
        sns.lineplot(x="budget",y="num_critic_for_reviews",data=df2)
        st.pyplot(fig2)
    if st.sidebar.checkbox('Budget vs Box office Log Distribution'):
        plt.title("Log Distribution of Box office vs Budget")
        fig2 = plt.figure(figsize = (45, 20))
        sns.distplot(np.log(df3.budget), label = 'Budget') 
        sns.distplot(np.log(df3.gross), label = "Box office")
        plt.xlabel("Log Gross Revenue")
        plt.legend()
        st.pyplot(fig2)
    if st.sidebar.checkbox('Budget vs Box office'):
        fig3 = plt.figure(figsize = (35, 15))
        plt.title("Box office vs budget")
        sns.scatterplot(x = 'budget', y = 'gross', label = "Budget vs Box office", alpha = 0.7, data = df3)
        #sns.scatterplot(x = 'budget', y = 'dom_gross', label = "Domestic", alpha = 0.7, data = df)
        plt.title("Budget vs Gross Revenue")
        plt.ylabel("Gross Revenue ($M)")
        plt.xlabel("Production Budget ($M)")
        plt.legend()
        # matplotlib's tight_layout function magically fit elements within the figure
        # so it does not get cropped when you save it out.
        plt.tight_layout()
        #plt.savefig("filename.png")
        st.pyplot(fig3)
    
#Prediction
elif vis == 'Prediction':    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header('Custom Movie 1')
        
        num_critic_for_reviews = st.slider(label='num_critic_for_reviews', min_value=0, max_value=int(df4_quant['num_critic_for_reviews'].max()))
        num_user_for_reviews = st.slider(label='num_user_for_reviews', min_value=0, max_value=int(df4_quant['num_user_for_reviews'].max()))
        num_voted_users = st.slider(label='num_voted_users', min_value=0, max_value=int(df4_quant['num_voted_users'].max()), step=10000)
        imdb_score = st.slider(label='imdb_score', min_value=1.0, max_value=10.0, step=0.1)
        duration = st.slider(label='duration (minutes)', min_value=30, max_value=360)
        budget = st.slider(label='budget (USD)', min_value=10000000, max_value=int(df4_quant['budget'].max()), step=10000000)
        facenumber_in_poster = st.slider(label='facenumber_in_poster', min_value=int(df4_quant['facenumber_in_poster'].min()), max_value=int(df4_quant['facenumber_in_poster'].max()))
        director_facebook_likes = st.slider(label='director_facebook_likes', min_value=0, max_value=int(df4_quant['director_facebook_likes'].max()), step=1000)
        movie_facebook_likes = st.slider(label='movie_facebook_likes', min_value=0, max_value=int(df4_quant['movie_facebook_likes'].max()), step=1000)
        actor_1_facebook_likes = st.slider(label='actor_1_facebook_likes', min_value=0, max_value=int(df4_quant['actor_1_facebook_likes'].max()), step=1000)
        actor_2_facebook_likes = st.slider(label='actor_2_facebook_likes', min_value=0, max_value=int(df4_quant['actor_2_facebook_likes'].max()), step=1000)
        actor_3_facebook_likes = st.slider(label='actor_3_facebook_likes', min_value=0, max_value=int(df4_quant['actor_3_facebook_likes'].max()), step=1000)
        cast_total_facebook_likes = st.slider(label='cast_total_facebook_likes', min_value=0, max_value=int(df4_quant['cast_total_facebook_likes'].max()), step=10000)

        test_movie = np.array([
            num_critic_for_reviews,
            duration,
            director_facebook_likes,
            actor_3_facebook_likes,
            actor_1_facebook_likes,
            num_voted_users,
            cast_total_facebook_likes,
            facenumber_in_poster,
            num_user_for_reviews,
            budget,
            actor_2_facebook_likes,
            imdb_score,
            movie_facebook_likes
        ])
        test_movie = test_movie.reshape(1, -1)
        test_movie_scaled = scaler.transform(test_movie)
        pred = lasso_model.predict(test_movie_scaled)
        str_pred = '${:,.2f}'
        str_pred = str_pred.format(pred[0])
        st.write('Predicted Gross for your Movie:')
        st.write(str_pred)
        
    with col2:
        st.header('Custom Movie 2')
        
        num_critic_for_reviews = st.slider(label='num_critic_for_reviews', min_value=0, max_value=int(df4_quant['num_critic_for_reviews'].max()), key=2)
        num_user_for_reviews = st.slider(label='num_user_for_reviews', min_value=0, max_value=int(df4_quant['num_user_for_reviews'].max()), key=2)
        num_voted_users = st.slider(label='num_voted_users', min_value=0, max_value=int(df4_quant['num_voted_users'].max()), step=10000, key=2)
        imdb_score = st.slider(label='imdb_score', min_value=1.0, max_value=10.0, step=0.1, key=2)
        duration = st.slider(label='duration (minutes)', min_value=30, max_value=360, key=2)
        budget = st.slider(label='budget (USD)', min_value=10000000, max_value=int(df4_quant['budget'].max()), step=10000000, key=2)
        facenumber_in_poster = st.slider(label='facenumber_in_poster', min_value=int(df4_quant['facenumber_in_poster'].min()), max_value=int(df4_quant['facenumber_in_poster'].max()), key=2)
        director_facebook_likes = st.slider(label='director_facebook_likes', min_value=0, max_value=int(df4_quant['director_facebook_likes'].max()), step=1000, key=2)
        movie_facebook_likes = st.slider(label='movie_facebook_likes', min_value=0, max_value=int(df4_quant['movie_facebook_likes'].max()), step=1000, key=2)
        actor_1_facebook_likes = st.slider(label='actor_1_facebook_likes', min_value=0, max_value=int(df4_quant['actor_1_facebook_likes'].max()), step=1000, key=2)
        actor_2_facebook_likes = st.slider(label='actor_2_facebook_likes', min_value=0, max_value=int(df4_quant['actor_2_facebook_likes'].max()), step=1000, key=2)
        actor_3_facebook_likes = st.slider(label='actor_3_facebook_likes', min_value=0, max_value=int(df4_quant['actor_3_facebook_likes'].max()), step=1000, key=2)
        cast_total_facebook_likes = st.slider(label='cast_total_facebook_likes', min_value=0, max_value=int(df4_quant['cast_total_facebook_likes'].max()), step=10000, key=2)

        test_movie = np.array([
            num_critic_for_reviews,
            duration,
            director_facebook_likes,
            actor_3_facebook_likes,
            actor_1_facebook_likes,
            num_voted_users,
            cast_total_facebook_likes,
            facenumber_in_poster,
            num_user_for_reviews,
            budget,
            actor_2_facebook_likes,
            imdb_score,
            movie_facebook_likes
        ])
        test_movie = test_movie.reshape(1, -1)
        test_movie_scaled = scaler.transform(test_movie)
        pred = lasso_model.predict(test_movie_scaled)
        str_pred = '${:,.2f}'
        str_pred = str_pred.format(pred[0])
        st.write('Predicted Gross for your Movie:')
        st.write(str_pred)
    