#Importing libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def exec_script():

    #Just used as an effect to see the docker working - TODO remove
    print("entered exec_script")

    #Setting dataframe
    pd.set_option('display.max_colwidth', 120)
    df = pd.read_json('reviews.json')

    #Changing value names from the column sentiment
    df = df.replace({'sentiment' : { 'pos' : "Positive", 'neg' : "Negative", 'neutral' : "Neutral" }})

    #Spliting train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(df.text, df.sentiment, test_size = 0.2, random_state = 0)

    #TfidfVectorizer() - Convert a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer 
    #                    (Convert a collection of text documents to a matrix of token counts) followed by TfidfTransformer
    #                    (Transform a count matrix to a normalized tf or tf-idf representation).
    #LinearSVC() - Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, 
    #              so it has more flexibility in the choice of penalties and loss functions and should scale better to large 
    #              numbers of samples.
    pipe = Pipeline([
        ("tdidf", TfidfVectorizer()),
        ("clf", LinearSVC())])

    #Fitting to the model ("learning" the model)
    pipe.fit(x_train, y_train)

    #The predict() function works on top of the trained model and makes use of the learned label to map and predict the labels 
    #for the data to be tested.
    y_pred = pipe.predict(x_test)

    #Creating a data frame with our model trained and the predictions
    data_pred = pd.DataFrame({'Data Model':y_pred})

    #Saving the model as a .csv file
    data_pred.to_csv('data-model-docker.csv', index = False)

    #Just used as an effect to see the docker working - TODO remove
    print("csv saved")

def __init__():

    #Just used as an effect to see the docker working - TODO remove
    print("__init__ initialized")

    exec_script()

