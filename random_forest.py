import pandas as pd
from preprocessing import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
rf_params ={'criterion': ['entropy'],
            'max_depth': [8, 10, None],
            'min_samples_leaf': [7, 9],
            'min_samples_split': [30, 60],
            'class_weight' : ['balanced']}
df = pd.read_csv('../datasets/dataset.csv',sep=';')
X_train, X_test, y_train, y_test=preprocessing(df)
clf = GridSearchCV(RandomForestClassifier(), rf_params, n_jobs=-1, cv=10, scoring='f1_weighted')
clf.fit(X_train, y_train)
print('score=',clf.score(X_train, y_train))
print("Best model random forest")
print(clf.best_params_)
# Enregistrer les meileurs hyperparamètres dans un fichier
f=open(path_hyperparametre + 'random_forest_hyp',"wb")
pickle.dump(clf.best_params_, f)
f.close()