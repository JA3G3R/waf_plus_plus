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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

imputer = SimpleImputer(strategy="most_frequent")

ngrams = [' ', '!', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
        '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
        '=', '>', '?', '@', '[', '\\', '\]', '_', '`', 'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

vocabulary = {}
for i in range(len(ngrams)):
    vocabulary[ngrams[i]]=i

vectorizer_get = TfidfVectorizer(min_df=1,ngram_range=(1,1),analyzer='char',vocabulary=vocabulary)
vectorizer_post = TfidfVectorizer(min_df=1,ngram_range=(1,1),analyzer='char',vocabulary=vocabulary)


df = pd.read_csv('C:\\Users\\bhavarth\\OneDrive\\Desktop\\Project Exhibition\\datasets\\HTTP Requests\\csic_ecml_normalized_final.csv')

df = df[ (df.Method=='GET')|(df.Method == 'POST') ]
df = df.drop('Host-Header',axis=1)

get_df = df[df.Method == 'GET']
get_df=get_df.drop(['POST-Data'],axis=1)

post_df = df[df.Method == 'POST']
post_df=post_df.drop(['GET-Query'],axis=1)

post_df_imputed = pd.DataFrame(imputer.fit_transform(post_df))
post_df_imputed.columns = post_df.columns

get_df_imputed = pd.DataFrame(imputer.fit_transform(get_df))
get_df_imputed.columns = get_df.drop(['Content-Type'],axis=1).columns

post_temp = post_df_imputed.copy()
post_temp = post_temp.drop(['Class','Method'],axis=1)
get_temp = get_df_imputed.copy()
get_temp = get_temp.drop(['Class','Method'],axis=1)

post_df_imputed['Request'] =post_temp.agg(func=''.join,axis=1)
post_df_imputed.Request = post_df_imputed['Request'].apply(lambda d : d.lower())
get_df_imputed['Request'] = get_temp.agg(func=''.join,axis=1)
get_df_imputed.Request = get_df_imputed['Request'].apply(lambda d : d.lower())

print(post_df_imputed.Request.head())

post_df_train,post_df_test,y_post_train,y_post_test = train_test_split(post_df_imputed.drop('Class',axis=1),post_df_imputed.Class,test_size=0.2,random_state=11)
get_df_train,get_df_test,y_get_train,y_get_test = train_test_split(get_df_imputed.drop('Class',axis=1),get_df_imputed.Class,test_size=0.2,random_state=11)

post_df_test_tmp = post_df_test.copy()
post_df_test_tmp['Class'] = y_post_test
post_df_test_tmp.to_csv('post_test.csv')

get_df_test_tmp = get_df_test.copy()
get_df_test_tmp['Class'] = y_get_test
get_df_test_tmp.to_csv('get_test.csv')

post_matrix=vectorizer_post.fit_transform(post_df_train.Request)
get_matrix=vectorizer_get.fit_transform(get_df_train.Request)
features_train_post=pd.DataFrame(post_matrix.todense(),columns=vectorizer_post.get_feature_names_out())
pickle.dump(vectorizer_post,open('models/tfidf_post.sav','wb'))
features_train_get = pd.DataFrame(get_matrix.todense(),columns=vectorizer_get.get_feature_names_out())
pickle.dump(vectorizer_get,open('models/tfidf_get.sav','wb'))


y_get_train = y_get_train.replace(to_replace={"Anomalous":0,"Valid":1})
y_post_train = y_post_train.replace(to_replace={"Anomalous":0,"Valid":1})
y_get_test = y_get_test.replace(to_replace={"Anomalous":0,"Valid":1})
y_post_test = y_post_test.replace(to_replace={"Anomalous":0,"Valid":1})

svc = SVC(C=5,kernel='rbf',random_state=11)
linear_svc = LinearSVC(C=6,random_state=11)

#logistitc regression

lr_get = LogisticRegression(class_weight='balanced',random_state=11,max_iter=250)
lr_post = LogisticRegression(class_weight='balanced',random_state=11,max_iter=250)

# lr_get.fit(X1_get,y1_get)
# lr_post.fit(X1_post,y1_post)
# pickle.dump(lr_get,open('code/models/logistic_regression_get.sav','wb'))
# pickle.dump(lr_post,open('code/models/logistic_regression_post.sav','wb'))

# Linear SVC

# print("error with 1-gram Using LinearSVC for GET dataset: ",(err_lsvc_get:=(-100*sum(l:=cross_val_score(linear_svc,X1_get,y1_get,cv=5,scoring='neg_mean_absolute_error'))/len(l))))
# err_lsvc_get = str(err_lsvc_get)[:4]
# pickle.dump(linear_svc,open(f'models/linear_svc_get-{err_lsvc_get}.sav','wb'))

# print("error with 1-gram Using LinearSVC for POST dataset: ",(err_lsvc_post:=(-100*sum(l:=cross_val_score(linear_svc,X1_post,y1_post,cv=5,scoring='neg_mean_absolute_error'))/len(l))))
# err_lsvc_post = str(err_lsvc_post)[:4]
# pickle.dump(linear_svc,open(f'models/linear_svc_post-{err_lsvc_post}.sav','wb'))

# SVC 

# print("error with 1-gram Using normal SVC with POST: ",(err_svc_post:=(-100*(sum(l:=cross_val_score(svc,X1_post,y1_post,cv=5,scoring='neg_mean_absolute_error'))/len(l)))))
matrix_post = vectorizer_post.transform(post_df_test.Request)
matrix_get = vectorizer_get.transform(get_df_test.Request)
features_test_post=pd.DataFrame(matrix_post.todense(),columns=vectorizer_post.get_feature_names_out())
features_test_get = pd.DataFrame(matrix_get.todense(),columns=vectorizer_get.get_feature_names_out())

svc.fit(features_train_post,y_post_train)
preds=svc.predict(features_test_post)
print("ERROR with POST model: ",mean_absolute_error(y_post_test,preds))
pickle.dump(svc,open(f'models/normal_svc_post.sav','wb'))

# print("error with 1-gram Using normal SVC with GET dataset: ",(err_svc_get:=(-100*(sum(l:=cross_val_score(svc,X1_get,y1_get,cv=5,scoring='neg_mean_absolute_error'))/len(l)))))
# err_svc_get=str(err_svc_get)[:4]

svc.fit(features_train_get,y_get_train)
preds=svc.predict(features_test_get)
print("ERROR with GET model: ",mean_absolute_error(y_get_test,preds))
pickle.dump(svc,open(f'models/normal_svc_get.sav','wb'))

# print("error with 1-gram Using Logistic Regression: ",(err_lr:=(-100*(sum(l:=cross_val_score(lr,X1,y1,cv=5,scoring='neg_mean_absolute_error'))/len(l)))))
# err_lr= str(err_lr)[:4]
# pickle.dump(linear_svc,open(f'logistic_regression_{err_lr}.sav','wb'))
# print("error with 1-gram Using Elliptic Envelope: ",(err_ee:=(-100*(sum(l:=cross_val_score(ee,X1,y1,cv=5,scoring='neg_mean_absolute_error'))/len(l)))))
# err_ee= str(err_ee)[:4]
# pickle.dump(linear_svc,open(f'elliptic_envelop_{err_ee}.sav','wb'))

# print("Accuracy with 1-gram Using SVM: ",-1*(sum(l:=cross_val_score(svm,X1,y1,cv=5,scoring='accuracy'))/len(l)))
# print("Accuracy with 1-gram Using Isolation Forest: ",-1*(sum(m:=cross_val_score(_if,X1,y1,cv=5,scoring='accuracy'))/len(m)))

# print("Accuracy with 2-gram Using Isolation Forest : ",-1*(sum(n:=cross_val_score(_if,X2,y2,cv=5,scoring='accuracy'))/len(n)))
# print("Accuracy with 2-gram Using SVM: ",-1*(sum(o:=cross_val_score(svm,X2,y2,cv=5,scoring='accuracy'))/len(o)))
