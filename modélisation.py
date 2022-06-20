import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from preprocessing import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
import pickle
path_hyperparametre="../hyperparametre/"
path_images = '../images/'

df = pd.read_csv('../datasets/dataset.csv',sep=';')
X_train, X_test, y_train, y_test=preprocessing(df)
# Cette fonction import les hyperparamètres des modèles
def hyperparametre_model(liste_hyp: list):
    #global xgb_best_param
    for element in liste_hyp:
        if element=='regression_logistique_hyp':
            f=open(path_hyperparametre + element,"rb")
            lr_best_param=pickle.load(f)
            lr_best_param['class_weight']=lr_best_param.pop('lr__class_weight')
            lr_best_param['max_iter']=lr_best_param.pop('lr__max_iter')
            lr_best_param['solver']=lr_best_param.pop('lr__solver')
        elif element=='random_forest_hyp':
            f=open(path_hyperparametre + element,"rb")
            rf_best_param=pickle.load(f)
        elif element=='Xgboost_hyp':
            f=open(path_hyperparametre + element,"rb")
            xgb_best_param=pickle.load(f)
        else :
            f=open(path_hyperparametre + element,"rb")
            svm_best_param=pickle.load(f)
            svm_best_param['C']=svm_best_param.pop('svm__C')
            svm_best_param['class_weight']=svm_best_param.pop('svm__class_weight')
            svm_best_param['kernel']=svm_best_param.pop('svm__kernel')
    return lr_best_param,rf_best_param,xgb_best_param,svm_best_param

liste_hyp=['regression_logistique_hyp','random_forest_hyp','Xgboost_hyp','SVM_hyp']
lr_params,rf_params,xgb_params,svm_params=hyperparametre_model(liste_hyp)

models = {
    'lr':LogisticRegression(**lr_params),
    'rf':RandomForestClassifier(**rf_params),
    'xgb':xgb.XGBClassifier(**xgb_params),
    'svm':SVC(**svm_params)
}

model_abrv = {
    'lr':'Logistic Regression',
    'rf':'Random Forest Classifier',
    'xgb':'XGB Classifier',
    'svm':'SVM Classifier'
}

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, model='clf', save=True):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a seaborn heatmap.
     Saves confusion matrix file to jpg file."""
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap=plt.cm.Oranges)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(model_abrv[model])
    plt.tight_layout()
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    if save == True:
        plt.savefig(path_images +'tuned_' + model_abrv[model] + '_confusion_matrix.jpg')
    #plt.show()

def model(clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models, save=False, print_stat=True, inc_train=False, cv=False):
    clf_model = models[clf]
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)
    if print_stat == True:
        clf_report = pd.DataFrame(classification_report(y_test,y_pred, output_dict=True)).T
        clf_report.to_csv(path_images +'tuned_' + model_abrv[clf] + '_classification_report.csv')
        print(model_abrv[clf])
        print('\nTest Stats\n', classification_report(y_test,y_pred))
        print(f"accuracy = {accuracy_score(y_test, y_pred)}")
        print_confusion_matrix(confusion_matrix(y_test, y_pred), unique_labels(y_test, y_pred), model=clf)
        if inc_train == True:
            print(model_abrv[clf])
            print('\nTrain Stats\n', classification_report(y_train,clf_model.predict(X_train)))
            print_confusion_matrix(confusion_matrix(y_train, clf_model.predict(X_train)), unique_labels(y_test, y_pred), model=clf)
    if cv == True:
        print(model_abrv[clf] + ' CV Accuracy:',
              np.mean(cross_val_score(clf_model, X_train, y_train, cv=10, scoring='accuracy')))
    if save == True:
        return clf_model

for key in models.keys():
    model(key, inc_train=False)

clf_final=RandomForestClassifier(**rf_params)
model = clf_final.fit(X_train, y_train)
feature_imp = pd.Series(model.feature_importances_,index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10,20))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Score importance des variables')
plt.ylabel('Features')
plt.title("Importance des variables random forest")
plt.legend()
plt.savefig(path_images + 'features_importances_random_forest.png')
plt.show()