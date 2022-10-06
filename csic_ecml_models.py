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

vectorizer_1 = TfidfVectorizer(min_df=1,ngram_range=(1,1),analyzer='char',vocabulary=vocabulary)
vectorizer_2 = TfidfVectorizer(min_df=1,ngram_range=(2,2),analyzer='char')


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

post_df_imputed['Request'] = post_df_imputed.agg(func=' '.join,axis=1)
post_df_imputed.Request = post_df_imputed['Request'].apply(lambda d : d.lower())
get_df_imputed['Request'] = get_df_imputed.agg(func=' '.join,axis=1)
get_df_imputed.Request = get_df_imputed['Request'].apply(lambda d : d.lower())


features_1_post=pd.DataFrame(vectorizer_1.fit_transform(post_df_imputed.Request).todense(),columns=vectorizer_1.get_feature_names_out())
features_1_get = pd.DataFrame(vectorizer_1.fit_transform(get_df_imputed.Request).todense(),columns=vectorizer_1.get_feature_names_out())
# features_2=pd.DataFrame(vectorizer_2.fit_transform(df_imputed.Request).todense(),columns=vectorizer_2.get_feature_names_out())

print(features_1_get.columns)
print(features_1_post.columns)
print("1-gram features shape for post requests ",features_1_post.shape)
print("1-gram features shape for get requests ",features_1_get.shape)

features_1_post['Class'] = post_df_imputed.Class
features_1_post['Class'] = post_df_imputed.Class.replace(to_replace={"Anomalous":0,"Valid":1})

features_1_get['Class'] = get_df_imputed.Class
features_1_get['Class'] = get_df_imputed.Class.replace(to_replace={"Anomalous":0,"Valid":1})

X1_get = features_1_get
y1_get = X1_get.pop('Class')
X1_post = features_1_post
y1_post = X1_post.pop('Class')

X1_post_train,X1_post_test,y1_post_train,y1_post_test = train_test_split(X1_post,y1_post,test_size=0.2,random_state=11)


svc = SVC(C=15,kernel='rbf',random_state=11)
linear_svc = LinearSVC(C=6,random_state=11)

#logistitc regression

lr_get = LogisticRegression(class_weight='balanced',random_state=11,max_iter=250)
lr_post = LogisticRegression(class_weight='balanced',random_state=11,max_iter=250)

# lr_get.fit(X1_get,y1_get)
# lr_post.fit(X1_post,y1_post)
# pickle.dump(lr_get,open('code/models/logistic_regression_get.sav','wb'))
# pickle.dump(lr_post,open('code/models/logistic_regression_post.sav','wb'))

# Linear SVC

print("error with 1-gram Using LinearSVC for GET dataset: ",(err_lsvc_get:=(-100*sum(l:=cross_val_score(linear_svc,X1_get,y1_get,cv=5,scoring='neg_mean_absolute_error'))/len(l))))
# err_lsvc_get = str(err_lsvc_get)[:4]
# pickle.dump(linear_svc,open(f'models/linear_svc_get-{err_lsvc_get}.sav','wb'))

# print("error with 1-gram Using LinearSVC for POST dataset: ",(err_lsvc_post:=(-100*sum(l:=cross_val_score(linear_svc,X1_post,y1_post,cv=5,scoring='neg_mean_absolute_error'))/len(l))))
# err_lsvc_post = str(err_lsvc_post)[:4]
# pickle.dump(linear_svc,open(f'models/linear_svc_post-{err_lsvc_post}.sav','wb'))

# SVC 

# print("error with 1-gram Using normal SVC with POST: ",(err_svc_post:=(-100*(sum(l:=cross_val_score(svc,X1_post,y1_post,cv=5,scoring='neg_mean_absolute_error'))/len(l)))))

# svc.fit(X1_post,y1_post)
# pickle.dump(svc,open(f'models/normal_svc_post.sav','wb'))

# print("error with 1-gram Using normal SVC with GET dataset: ",(err_svc_get:=(-100*(sum(l:=cross_val_score(svc,X1_get,y1_get,cv=5,scoring='neg_mean_absolute_error'))/len(l)))))
# err_svc_get=str(err_svc_get)[:4]

# svc.fit(X1_get,y1_get)
# pickle.dump(svc,open(f'models/normal_svc_get.sav','wb'))

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
