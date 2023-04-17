
# Spotify Recommendtion System - Implementation on Streamlit - Capstone Project 

I started doing this project because I wanted to understand how recommendation systems work for music recommendation. 
I used a popular data set on Spotify data from Kaggle. I have demonstrated my skills in EDA, Data Wrangling, a few ML techniques like Kmeans, Decision trees, Random Forest, XGBoost. 
I even implemented a simple song recommendation system . In the end I decided to implement the entire project as an app on 
Streamlit IO.


## Demo

Link to streamlit - https://adithyasam-ganj-spotify-recommendation-cap-streamlit-app-5ilkr1.streamlit.app/


The newest version of the code does not include Djongo authentication for the streamlit app as it was removed so anyone could access it 
## Files

* The major portion of the analysis is done on the data_p.csv file available in Data folder . 
* Spotify_Recommendation_Capston_Project.ipynb: Jupyter notebook with EDA and Data wrangling 
* Streamlit_app.py - Contains the code for Streamlit app.

## Project 

1. EDA  - First step is to explore the data you are working with for the project. Dive in to see what you can find. There are some basic, required questions to be answered about the data you are working with throughout the rest of the notebook. Use this space to explore, before you dive into the details.
2. Feature Selection -  I was able to performe the necessary feature engineering on the selected features so that the model works without any bias. 
3. ML Modelling  for song popularity - I have used Decision Trees, Random Forest and XGBoost methods to predict the songs popularity. I tuned the hyperparameters with Grid Search and Randomised search cross validation. 
4. Song Recommendation System - Built a content based filtering algorithm. 

## Results 

Results are as follows: 

1.  I have initially run Linear Regression, KNeighboursRegressor, RandomForestRegressor, GradientBoostedRegressor and DecisionTreeRegressor. 
Decision tree regressor seemed to have the lowest r2 score of 0.68. RandomForestRegressor and GradientBoostedRegressor had the highest r2 score of 0.82 and 0.83 respectiverly. 

2. After proper hyperparameter tunning with Grid Search CV I could increase the r2 score of DecisionTreeRegressor to 0.816.

3. After hyperparameter tuning of random forest I achieved a r2 score for test data to be 0.829 and 0.915 for train data. 

4. I performed GradientBoostedRegressor with XGBoost and got a r2 score of 0.829 and 0.856 for test and train data respectively. 
