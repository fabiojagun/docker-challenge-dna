import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def exec_script():
    print("entrou no exec_script")
    pd.set_option('display.max_colwidth', 120)
    df = pd.read_json('reviews.json')

    df = df.replace({'sentiment' : { 'pos' : "Positive", 'neg' : "Negative", 'neutral' : "Neutral" }})

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

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)

    data_pred = pd.DataFrame({'Data Model':y_pred})

    data_pred.to_csv('data-model-docker.csv', index = False)
    print("salvou csv")

def save_google_drive():
    print("entrou no save_google_drive")
    gauth = GoogleAuth()           
    drive = GoogleDrive(gauth)

    upload_file_list = ['data-model-docker.csv']
    for upload_file in upload_file_list:
	    gfile = drive.CreateFile({'parents': [{'id': '1wuqAacbFGDoP4wVh3iZpZS-LCu3M3TZT'}]})
	    # Read file and set it as the content of this instance.
	    gfile.SetContentFile(upload_file)
	    gfile.Upload() # Upload the file.

def __init__():
    print("entrou no __init__")
    exec_script()
    save_google_drive()

