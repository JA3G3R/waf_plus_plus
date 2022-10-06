import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df_xss = pd.read_csv("C:\\Users\\bhavarth\\OneDrive\\Desktop\\Project Exhibition\\datasets\\xss\\XSS_Dataset.csv")
# df_sqli1 = pd.read_csv("C:\\Users\\bhavarth\\OneDrive\\Desktop\\Project Exhibition\\datasets\\SQLi\\sqli.csv",encoding='utf-16')
# df_sqli2 = pd.read_csv("C:\\Users\\bhavarth\\OneDrive\\Desktop\\Project Exhibition\\datasets\\SQLi\\sqliv2.csv",encoding="utf-16")
# df_sqli3= pd.read_csv("C:\\Users\\bhavarth\\OneDrive\\Desktop\\Project Exhibition\\datasets\\SQLi\\SQLiV3.csv",encoding="utf-16")

# df_sqli = df_sqli1.merge(df_sqli2,on=['Sentence,Label'])
# df_sqli = df_sqli.merge(df_sqli3,on=['Sentence','Label'])

# print(df_sqli.head(10))
df_xss_imputed = pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(df_xss))
df_xss_imputed.columns = df_xss.columns

label = df_xss_imputed.pop('Label')
ngrams = [' ', '!', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
'/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
'=', '>', '?', '@', '[', '\\', '\]', '_', '`', 'a', 'b', 'c', 'd', 'e',
'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

vocabulary = {}
for i in range(len(ngrams)):
    vocabulary[ngrams[i]] = i
vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,1),vocabulary=vocabulary)

features_1 = pd.DataFrame(vectorizer.fit_transform(df_xss_imputed.Sentence).todense(),columns=vectorizer.get_feature_names())
svc = SVC(C=6,kernel='rbf',random_state=11)
# print(df_xss_imputed.Label.head())
svc.fit(features_1,label.astype('int'))
pickle.dump(svc,open('code/models/xss_svc.sav','wb'))
