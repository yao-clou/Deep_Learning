import pandas as pd
import numpy as np
import re
import seaborn as sns
from sklearn.model_selection import train_test_split

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)
# Chargement des données

def transformation(df):
    col=['interet_compte_epargne_total','espace_client_web']
    df[col[0]]=pd.to_numeric(df[col[0]],errors='coerce')
    df[col[1]]=df[col[1]].astype('object')
    return df

def encodage(df):
    cat_cols = []
    var_features = [feature for feature in df.columns if feature.startswith('var_')]
    other_features = [feature for feature in df.columns if feature not in var_features ]
    for column in other_features:
        if column not in ['id_client','anciennete_mois','agios_6mois','interet_compte_epargne_total','age','churn']:
            cat_cols.append(column)
    # Encodage one-hot
    df= pd.get_dummies(df,columns=cat_cols,dummy_na=True)
    nan_df = df.loc[:, df.columns.str.endswith("_nan")] # extraire les variables se terminant par _nan
    # Cette partie du code reordonne les variables _nan
    pattern = "^([^_]*)_"
    regex = re.compile(pattern)
    for index in df.index:
        for col_nan in nan_df.columns:
            if df.loc[index,col_nan] == 1:
                col_id = regex.search(col_nan).group(1)
                targets = df.columns[df.columns.str.startswith(col_id+'_')]
                df.loc[index, targets] = np.nan
    df.drop(df.columns[df.columns.str.endswith('_nan')], axis=1, inplace=True)
    return df

def imputation(df):
    # Remplacer les valeurs manquantes par 0
    df=df.fillna(0)
    # suppression de colonnes
    var_features=['var_25','var_26','var_27','var_9','var_11','var_13','var_15','var_35','var_17','var_36','var_19','var_38']
    cols_to_delete = ['id_client']
    cols_to_delete.extend(var_features)
    #df=df.drop(['id_client'],axis=1)
    df=df.drop(cols_to_delete,axis=1)
    # Encoder la target variable churn
    code={'oui':1,'non':0}
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)
    return  df

def compute_IQR_find_outliers(df: pd.DataFrame):
    num_df=df[['age','anciennete_mois','agios_6mois','interet_compte_epargne_total']]
    num_cols = list(num_df.columns)
    num_df_var_=df[[feature for feature in df.columns if feature.startswith('var_')]]
    num_cols_var_ = list(num_df_var_.columns)
    col_numerique=num_cols + num_cols_var_
    #Calcul de l'interval interquartile et suppression des valeurs extreme

    Q1 = df[col_numerique].quantile(0.25)
    Q3 = df[col_numerique].quantile(0.75)
    IQR = Q3 - Q1
    IQR_min = Q1 - 1.5 * IQR
    IQR_max = Q3 + 1.5 * IQR

    df = df[~((df[col_numerique] < IQR_min) | (df[col_numerique] > IQR_max)).any(axis=1)]

    return df

# Cette fonction renvoie 4 variables
def preprocessing(df):
    df = transformation(df)
    df = encodage(df)
    df = compute_IQR_find_outliers(df)
    #df = feature_engineering(df)
    df = imputation(df)
    X = df.drop('churn', axis=1)
    y = df['churn']
    #separer les données en train set (80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

