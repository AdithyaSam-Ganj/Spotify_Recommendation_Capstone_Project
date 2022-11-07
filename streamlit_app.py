import streamlit as st 

#import streamlit as st

# Using object notation
import numpy as np
import pandas as pd

#Visualisation
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# to see correlation of each variable with a particular target
from yellowbrick.target import FeatureCorrelation

#Progreebar
from tqdm import tqdm

#feather Files
import pyarrow.feather as feather

import time

# For transformations and predictions
from scipy.optimize import curve_fit
from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# For scoring
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score,mean_absolute_error

# For validation
from sklearn.model_selection import train_test_split

# Streamlit IO
import streamlit as st
from stqdm import stqdm

def read_data():
    ''' This is for reading all data files'''
    #df_artist = pd.read_csv('data/data_by_artist_p.csv')
    #df_gener = pd.read_csv('data_by_genres_p.csv')
    #df_year = pd.read_csv('data_by_year_p.csv')
    df = pd.read_csv('Data/data_p.csv')
    #df_w_gener = pd.read_csv('data_w_genres_p.csv')

    # Tried to group by artists and look at data. but it is very confusing because of the way it is arranged.
    # trying to drop  [ ] '
    df['artists']=df['artists'].str.replace('[','')
    df['artists']=df['artists'].str.replace(']','')
    df['artists']=df['artists'].str.replace("'","")
    return df#, df_artist #,df_gener,df_year,df_w_gener

## Display Picture on webpage 

# Display the corr plot of the file
def corr_plot(ds):
    fig,ax = plt.subplots()
    plt.figure(figsize=(20,10))
    #sns.set(style="whitegrid")
    corr = ds.corr()
    sns.heatmap(corr,cmap='BrBG_r', ax=ax)
    ax.set_title('A Heat map of all the song data that is stored in Spotify ')
    st.write(fig)

# List of the most popular tracks
def popular_tracks(ds, number ):
    fig, axis = plt.subplots(figsize = (10,7))
    popular = ds.groupby("name")['popularity'].mean().sort_values(ascending=False).head(number)
    axis = sns.barplot(x = popular, y = popular.index, palette="mako",orient = 'h')
    axis.set_title('Top 15 Popular Tracks')
    axis.set_ylabel('Tracks')
    axis.set_xlabel('Popularity')
    plt.xticks(rotation = 90)
    st.write(fig)

# List of Popular Artist
def popular_artists(ds, number ):
    fig, axis = plt.subplots(figsize = (10,7))
    popular = ds.groupby("artists")['popularity'].sum().sort_values(ascending=False)[:number]
    axis = sns.barplot(x = popular,y = popular.index,palette="mako",orient = 'h')
    axis.set_title('Top 20 Artists with Popularity')
    axis.set_ylabel('Popularity')
    axis.set_xlabel('Tracks')
    plt.xticks(rotation = 90)
    st.write(fig)

# Visualising the popularity of artists at that time
def Visualise_popularity(ds, artist_ka_name):
    Beatles = ds[ds['artists'] == artist_ka_name]
    #plt.rcParams['figure.figsize'] = (11,7)
    # line plot passing x,y
    fig, axis = plt.subplots()
    axis = sns.lineplot(x='year', y='popularity', data=Beatles, color='green')
    axis.set_title("The Beatles Popularity")
    axis.set_xlabel('Year')
    axis.set_ylabel('Popularity')
    #axis.show()
    st.write(fig)

###################################################################

# The Recommendation System part
def read_for_reco():
    df = pd.read_csv('Data/data_p.csv')
    # Apply Aritists class on train and test seperatly
    num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num = df.select_dtypes(include=num_types)
    # Normalize allnumerical columns so that min value is 0 and max value is 1
    for col in num.columns:
        normalize_column(df,col)
    df.drop(['release_date', 'id'], axis=1, inplace=True)
    return df


class Artists:
    """
     This transformer recives a DF with a feature 'artists' of dtype object
      and convert the feature to a float value as follows:
      1. Replace the data with the artists mean popularity
      2. Replace values where artists appear less than MinCnt with y.mean()
      3. Replace values where artists appear more than MaxCnt with 0
      PARAMETERS:
      ----------
      MinCnt (int): Minimal treshold of artisits apear in dataset, default = 3
      MaxCnt (int): Maximal treshold of artisits apear in dataset, default = 600
      RERTURN:
      ----------
      A DataFrame with converted artists str feature to ordinal floats
    """
    def __init__(self, MinCnt=3.0, MaxCnt=600.0):
        self.MinCnt = MinCnt
        self.MaxCnt = MaxCnt
        self.artists_df = None

    def fit(self, X, y):
        self.artists_df = y.groupby(X.artists).agg(['mean', 'count'])
        # print(self.artists_df.loc['unknown'])
        self.artists_df.loc['unknown'] = [y.mean(), 1]
        # print(self.artists_df.loc['unknown'])
        self.artists_df.loc[self.artists_df['count'] <= self.MinCnt, 'mean'] = y.mean()
        self.artists_df.loc[self.artists_df['count'] >= self.MaxCnt, 'mean'] = 0
        return self

    def transform(self, X, y=None):
        X['artists'] = np.where(X['artists'].isin(self.artists_df.index), X['artists'], 'unknown')
        X['artists'] = X['artists'].map(self.artists_df['mean'])
        return X


# Instrumental Transformer Criteria

def instrumental(X):
    X['instrumentalness'] = list(map((lambda x: 1 if x < 0.1 else (3 if x > 0.95 else 2)), X.instrumentalness))


class Tempo():
    """Eliminates Zero values from tempo columns and replace it
       with the median or mean of non-zero values as specified.
       defaut is set to 'median'.
    """
    def __init__(self, method='median'):
        self.method = method
    def transform(self, X):
        if self.method == 'median':
            X.loc[X['tempo'] == 0, 'tempo'] = X.loc[X['tempo'] > 0, 'tempo'].median()
        elif self.method == 'mean':
            X.loc[X['tempo'] == 0, 'tempo'] = X.loc[X['tempo'] > 0, 'tempo'].mean()
        else:
            raise Exception("Method can be 'median' or 'mean' only!")
        return X


def normalize_column(df, col):
    """
    col - column in the dataframe which needs to be normalized
    """
    max_d = df[col].max()
    min_d = df[col].min()
    df[col] = (df[col] - min_d) / (max_d - min_d)

class Song_Recommender():
    """
    Neighbourhood Based Collborative Filterng REcoomendation System using similarity Metrics
    Manhattan Distance is calculated for all songs and Recommend Songs that are similar to it based on any given song
    """
    def __init__(self, data):
        self.data_ = data

    # function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations(self, song_name, n_top , flag):
        distances = []
        if song_name == 'Select Song' :
            return None
        else :
            # choosing the given song_name and dropping it from the data
            song = self.data_[(self.data_.name.str.lower() == song_name.lower())].head(1).values[0]
            rem_data = self.data_[self.data_.name.str.lower() != song_name.lower()]
            for r_song in stqdm(rem_data.values):
                dist = 0
                for col in np.arange(len(rem_data.columns)):
                    # indeces of non-numerical columns(id,Release date,name,artists)
                    if not col in [3, 7, 13]:
                        # calculating the manhettan distances for each numerical feature
                        dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
                distances.append(dist)
            rem_data['distance'] = distances
            # sorting our data to be ascending by 'distance' feature
            rem_data = rem_data.sort_values('distance')
            columns = ['artists', 'name']
            if flag == 1:
                random_numbers = np.random.randint(30,size=n_top)
                random_numbers = random_numbers.tolist()
                rem_data = rem_data.reset_index()
                return rem_data.loc[random_numbers,columns]
            else :
                return rem_data[columns][:n_top]



def plotting_animation(ds,feature_):
    x_values = ds.groupby('year')[feature_].mean()
    chart = st.line_chart(data = x_values.iloc[0:1])#x = x_values, y = x_values.index,use_container_width=True )
    # sns.lineplot(x.index, x, label=col)
    # plt.figure(figsize=(15, 5))
    # sns.set_style("whitegrid")
    for i in range(len(x_values)):
        to_plot = x_values.iloc[i:i+1]
        chart.add_rows(to_plot)
        time.sleep(0.02)


###################################################################################################### Streamlit code
add_sidebar = st.sidebar.selectbox(
    "What would you like to view?",
    ("Spotify in Graphs", "Try our Song Recommendation System")
)

if add_sidebar == "Spotify in Graphs":
    st.header('Spotify in Graphs')
    st.write('With 60,000 songs being uploaded into spotify per day the total number of songs are truly massive.'
             'In this data set we are doing to look at only the top 2000 songs from each year from 1960 - 2020. ')
    df = read_data()# , df_artist = read_data()

    features = ["", "acousticness", "danceability", "energy", "speechiness", "liveness", "valence"]

    st.write("Spotify stores data about some key attributes of the songs ")
    feature = st.selectbox('Choose from a list of features', features)

    if feature == "":
        st.write("Choose a Features")
    else:
        plotting_animation(df, feature)
        if feature == "acousticness" :
            st.write("**Acoustiness**- A confidence measure from 0.0 to 1.0 of whether the track "
                     "is acoustic. 1.0 represents high confidence the track is acoustic.")
        elif feature == "danceability" :
            st.write("**Danceability** — Danceability describes how suitable a track is for dancing based on a combination "
                     "of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is the most danceable.")
        elif feature == "energy" :
            st.write("**Energy** — Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity "
                     "and activity. Typically, energetic tracks feel fast, loud, and noisy.")
        elif feature == "speechiness":
            st.write("**Speechiness** - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music."
                     " Values below 0.33 most likely represent music and other non-speech-like tracks")
        elif feature == "liveness":
            st.write("**Liveness** - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. "
                "A value above 0.8 provides strong likelihood that the track is live.")
        elif feature == "valence":
            st.write("Valence ** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric),"
                     " while tracks with low valence sound more negative (e.g. sad, depressed, angry).")
    
    st.write( "Other key attributes of songs"  )
    st.write("\n")
    st.write("**Key** — The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. Ex: 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.")
    st.write("\n")
    st.write("**Mode** — Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.")
    st.write("\n")
    st.write("**Popularity** — The popularity of the track. The value will be between 0 and 100, with 100 being the most popular.")
    st.write("\n")
    st.write("**Tempo** — The overall estimated tempo of a track in beats per minute (BPM).")
    st.write("\n")
    st.subheader('Graph showing the corellation between these key attributes')
    corr_plot(df)

    st.subheader('Here is a list of the most popular tracks(December 2020)')
    popular_tracks(df,10)

    st.subheader('Here is a list of the most popular artists(December 2020)')
    popular_artists(df,10)

    st.subheader('Choose from the list of artists')
    top_artist = ['The Beatles']
    top_artist = st.selectbox('Choose from some of the top all time favourite artists', top_artist)

    Visualise_popularity(df, top_artist)

    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()

    # progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.


###################################################################################################

if add_sidebar == "Try our Song Recommendation System":
    st.header('Song Recommendation System')
    df = read_for_reco()
    recommender = Song_Recommender(df)
    song_list = df['name'].unique()
    song_list_df = pd.DataFrame(song_list, columns=["song_name"])
    song_list_df.to_feather("song_list_df_feather.feather")
    song_list_df_feather = feather.read_feather("song_list_df_feather.feather")
    # s_l = song_list_df_feather.tolist()
    user_name = st.selectbox("Choose from list of songs " , song_list_df_feather , label_visibility = 'collapsed')
    number = st.number_input('Enter number of songs to suggest: ',min_value = 1, max_value = 20, step = 1)
    random = st.radio("Discover ?",('Yes', 'No'))
    if st.button('Suggest'):
        if random == 'Yes':
            st.write(recommender.get_recommendations(user_name, number, 1))
        else :
            st.write(recommender.get_recommendations(user_name, number, 0))
    else :
        st.write('waiting for input ')
######################################################################### Pakka code ^ 