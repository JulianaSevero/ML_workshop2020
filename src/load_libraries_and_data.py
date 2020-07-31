# data manipulation
import pandas as pd
import numpy as np
from operator import itemgetter


## learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

## preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


### model performance
from sklearn import metrics

#ploting modules
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

## Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV


xpath = './workspace/clinvarHC_RawVariables.csv' ; ypath = './workspace/clinvarHC_RawTarget.csv'
SEED = 2718281

X = pd.read_csv(xpath,sep = ',',index_col='ChrPosRefAlt')

y = pd.read_csv(ypath,sep = ',',header = None)
y.columns = ['ChrPosRefAlt','ClinvarHC']
y.set_index(['ChrPosRefAlt'],inplace = True) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=SEED,stratify = y)


categorical_feature_mask = X_train.dtypes==object       # esto nos da un vector booleano 
categorical_columns = X_train.columns[categorical_feature_mask].tolist()  # acá picnhamos los nombres de esas columnas
numerical_columns = X_train.columns[~X_train.columns.isin(categorical_columns)].tolist() # defino las numéricas como el complemento de las categóricas 


numerical_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(with_mean=False))])  # Esto es una vacancia de Sklearn, no permite aún "centrar" matrizes sparse


categorical_transformer = Pipeline(steps=[
    ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])      


preprocessor = ColumnTransformer(transformers = [
    ('num', numerical_transformer,numerical_columns),
    ('cat', categorical_transformer,categorical_columns)
])






def my_train_test_plot(gridsearch,grid,hyp,ax = None):
    sns.set_style("whitegrid")

    if ax == None:
        fig, ax = plt.subplots(1)        

    keys = grid[0]#.str.strplit('__')
    basename = [*keys][0].split('__')[0]
    name = basename+'__'+ hyp
    values = grid[0][name]    
        

    results = gridsearch.cv_results_
    cc = ['mean_train_score','std_train_score','mean_test_score']
    performance = pd.DataFrame(itemgetter(*cc)(results),index = cc,columns = values).transpose()

    performance[name] = [str(v) for v in values]

    perf_train = performance.mean_train_score
    perf_test = performance.mean_test_score

    plt.plot(performance[name], performance.mean_test_score)

    ylow =   perf_train - performance.std_train_score
    yup = perf_train + performance.std_train_score

    ax.fill_between(performance[name], ylow, yup,
        alpha=0.5, edgecolor='lightgray', facecolor='lightgray')

    ax.set_ylabel('ROC-AUC')
    ax.set_xlabel(hyp)
    ax.set_xticklabels(np.round(values,2), rotation=45)

