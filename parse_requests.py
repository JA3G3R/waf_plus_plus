import numpy as np # linear algebra
import pickle
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

with open('datasets/HTTP Requests/test requests/normalTrafficTraining.txt','r',encoding='utf-8') as f:
    data = f.readlines()
with open('datasets/HTTP Requests/test requests/anomalousTrafficTest.txt','r',encoding='utf-8') as f:
    anomalous = f.readlines()

anomalous = "".join(anomalous)
data = "".join(data)
final_get = []
final_get_class = []
final_post = []
final_post_class = []
label='N'
while len(data):
    data = data.lstrip('\n').lstrip(' ')
    print(len(data))
    if data[:4] == 'GET ':
        get= data.split('\n\n\n')[0]
        final_get.append(get)
        data = data[len(get):]
        if label=='A':
            final_get_class.append(0)
        elif label=='N':
            final_get_class.append(1)
    elif data[:4]== 'POST' or data[:4]=='PUT ':
        post = data.split('\n\n')[:2]
        post = "\n\n".join(post)
        if data[:4] != 'PUT ':
            final_post.append(post)
        data = data[len(post):]
        if label=='A':
            final_post_class.append(0)
        elif label=='N':
            final_post_class.append(1)
    if len(data)==0:
        if label!='A':
            print('Done Normal')
            data = anomalous
            label='A'
        else:
            print("Done Finally")

ngrams = [' ', '!', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
        '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
        '=', '>', '?', '@', '[', '\\', '\]', '_', '`', 'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

vocabulary = {}
for i in range(len(ngrams)):
    vocabulary[ngrams[i]]=i

vectorizer_1 = TfidfVectorizer(min_df=1,ngram_range=(1,1),analyzer='char',vocabulary=vocabulary)

features_1_post = pd.DataFrame(vectorizer_1.fit_transform(pd.Series(final_post)).todense(),columns=vectorizer_1.get_feature_names_out())
features_1_get = pd.DataFrame(vectorizer_1.fit_transform(pd.Series(final_get)).todense(),columns=vectorizer_1.get_feature_names_out())

svc_get = SVC(C=15,kernel='rbf',random_state=11)
svc_post = SVC(C=15,kernel='rbf',random_state=11)
svc_get.fit(features_1_get,final_get_class)
svc_post.fit(features_1_post,final_post_class)
pickle.dump(svc_get,open('code/models/svc_new_get.sav','wb'))
pickle.dump(svc_post,open('code/models/svc_new_post.sav','wb'))

linear_svc_get = LinearSVC(C=6,random_state=11)
linear_svc_post= LinearSVC(C=6,random_state=11)
linear_svc_post.fit(features_1_post,final_post_class)
linear_svc_get.fit(features_1_get,final_get_class)
pickle.dump(linear_svc_get,open('code/models/linear_svc_new_get.sav','wb'))
pickle.dump(linear_svc_get,open('code/models/linear_svc_new_get.sav','wb'))

lr_get = LogisticRegression(class_weight='balanced',random_state=11,max_iter=250)
lr_post = LogisticRegression(class_weight='balanced',random_state=11,max_iter=250)
lr_get.fit(features_1_get,final_get_class)
lr_post.fit(features_1_post,final_post_class)
pickle.dump(lr_get,open('code/models/lr_new_get.sav','wb'))
pickle.dump(lr_post,open('code/models/lr_new_post.sav','wb'))






    