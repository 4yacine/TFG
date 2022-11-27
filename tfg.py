# -*- coding: utf-8 -*-
"""
@author: yacine
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from datetime import datetime, timedelta
from scipy.stats import zscore
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, balanced_accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early *')
#warnings.catch_warnings()

def leer(ruta, nombre):
    #Leo los datos del csv en un dataframe
    df = pd.read_csv(ruta)

    #Obtengo el tiempo inicial expresado en timestamp (UTC)
    t_inicial = df.columns[0]
    t_inicial = float(t_inicial)
   
    #Paso el tiempo inicial de timestamp a datetime
    t_inicial = datetime.fromtimestamp(t_inicial)

    #Obtengo la frecuencia de muestreo
    frec_muestreo = df._get_value(0, 0, takeable = True)

    #Paso la frecuencia de muestreo a segundos
    frec_muestreo = 1 / frec_muestreo

    #Elimino la fila correspondiente a la frecuencia de muestreo
    df = df.drop([0], axis=0)

    #Renombro la columna
    df.columns = [nombre]

    #Calculo un vector con los tiempos de cada muestra
    #Preguntar a Francisco si el tiempo inicial es la primera muestra, o si es t_inicial+frec
    tiempos = []; t = t_inicial; filas = df.shape[0]
    for i in range(filas):
        tiempos.append(t)
        t += timedelta(seconds=frec_muestreo)
    
    #Añado el tiempo correspondiente a cada muestra
    # En a columna 0 con nombre TimeStamp del vector tiempos y no permito que
    #tra columna tenga el mismo nombre
    df.insert(0, "TimeStamp", tiempos, allow_duplicates=False)
   
    return df

def perform_resample(df, mResample="0.04S", verbose=False):
    df = df.resample(mResample).mean()
    nas = sum(df.isna().sum())
    if nas > 0:
        if verbose:
            print("NAs after resample (" + mResample + "): " + str(nas))
    df = df.interpolate()
    return df


def preparar_datos(rutas):
    #cargamos en un dataframe nuestro CSV y lo indexamos por la columna timestamp parseando las dates
    diadema = pd.read_csv(rutas[0], header=0,index_col="TimeStamp", parse_dates=True)
    diadema = diadema.drop(['AUX_RIGHT', 'Accelerometer_X','Accelerometer_Y',
                            'Accelerometer_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 
                            'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8',
                            'HSI_TP10','Battery', 'Mellow',	'Concentration'], 
                           axis=1)
    
    eda = leer(rutas[1], 'EDA')
    temp = leer(rutas[2], 'TEMP')
    hr = leer(rutas[3], 'HR')
    bvp = leer(rutas[4], 'BVP')
    
    df = pd.merge(temp, eda, how='outer', on=["TimeStamp"])
    df = pd.merge(hr, df, how='outer', on=["TimeStamp"])
    df = pd.merge(bvp, df, how='outer', on=["TimeStamp"])
    df = pd.merge(diadema, df, how='outer', on=["TimeStamp"])

    df.set_index('TimeStamp', inplace=True)
    df = df.sort_values(by="TimeStamp")
    
    pos_m1 = np.where(df['Elements']=='/Marker/1')[0]
    pos_m2 = np.where(df['Elements']=='/Marker/2')[0]
    pos_m3 = np.where(df['Elements']=='/Marker/3')[0]
    
    #Creo un vector de dataframe con lo que hay entre cada marca 2 y 3
    dfs = []
    for i in zip(pos_m2, pos_m3):
        dfs.append(df[i[0]+1:i[1]]) #Cojo justo la fila de después de la primera marca y la anterior a la de la última marca
    
    #Creo un vector de dataframe con lo que hay entre cada marca 4 y 5
    dfs2 = []
    for i in zip(pos_m1, pos_m3):
        dfs2.append(df[i[0]+1:i[1]]) #Cojo justo la fila de después de la primera marca y la anterior a la de la última marca
    
    #Alineo los datos de cada dataframe de los fragmentos intensos
    dfs_resampled = []
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(['Elements'], axis=1)
        dfs_resampled.append(perform_resample(dfs[i], '0.5S'))
    
    #Alineo los datos de cada dataframe de los vídeos completos
    dfs2_resampled = []
    for i in range(len(dfs2)):
        dfs2[i] = dfs2[i].drop(['Elements'], axis=1)
        dfs2_resampled.append(perform_resample(dfs2[i], '0.5S'))
        
    etiquetas = ["tristeza", "neutro", "alegria", "sorpresa", "miedo", "asco", "ira"]

    num_col = len(dfs[0].columns)
    datos = [] #Será una matriz donde cada fila será una instancia
    for i in range(len(dfs_resampled)):
        for row in dfs_resampled[i].itertuples():
            fila = []
            for j in range(1, num_col+1): #y aqui poner num_col -1 como consecuencia de lo de abajo
                fila.append(row[j]) #aqui hay qe poner row[j]+1 pa no pillar la fecha
    
            fila.append(etiquetas[i]) #Añado la etiqueta a la fila
            datos.append(fila) #Añado una fila a la matriz de datos
                
    num_col = len(dfs2[0].columns)
    datos2 = [] #Será una matriz donde cada fila será una instancia
    for i in range(len(dfs2_resampled)):
        for row in dfs2_resampled[i].itertuples():
            fila = []
            for j in range(1, num_col+1): #y aqui poner num_col -1 como consecuencia de lo de abajo
                fila.append(row[j]) #aqui hay qe poner row[j]+1 pa no pillar la fecha
    
            fila.append(etiquetas[i]) #Añado la etiqueta a la fila
            datos2.append(fila) #Añado una fila a la matriz de datos
   
    return datos, datos2, dfs_resampled[0]

def unir_datos(datosFin, datosAux):
    for i in range(len(datosAux)):
        datosFin.append(datosAux[i])
    
    return datosFin


#Obtengo los datos
datos1 = []; datos2 = []
for i in range(1,24):
    rutas = []
    rutas.append("muestras/sujeto" + str(i) + "/diadema.csv")
    rutas.append("muestras/sujeto" + str(i) + "/EDA.csv")
    rutas.append("muestras/sujeto" + str(i) + "/TEMP.csv")
    rutas.append("muestras/sujeto" + str(i) + "/HR.csv")
    rutas.append("muestras/sujeto" + str(i) + "/BVP.csv")
    d1, d2, df10 = preparar_datos(rutas)
    
    datos1 = unir_datos(datos1, d1)
    datos2 = unir_datos(datos2, d2)


# print("datos1", len(datos1))
# print("datos2", len(datos2))


###############################################################################
    #MODELOS PARA EL CONJUNTO DE DATOS REFERENTE A LOS FRAGMENTOS INTENSOS#
###############################################################################
print("\n")
print("MODELOS PARA EL CONJUNTO DE DATOS REFERENTE A LOS FRAGMENTOS INTENSOS")
print("\n")

#Obtengo las características y las etiquetas a partir de los datos
datos1 = np.array(datos1)
features1 = datos1[:,:-1]
target1 = datos1[:,-1]

#Separo los datos en test y training
X_train, X_test, y_train, y_test = train_test_split(features1, target1,
               test_size=0.2, random_state=0, shuffle=True, stratify=target1)

#Definción de los plots
sns.set_style("darkgrid")
sns.set_palette("pastel")
sns.despine()

#Almaceno todas las variables
var = ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9',
                'Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7',
                'Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8',
                'Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10',
                'RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10','BVP','HR','TEMP','EDA']

#Paso el conjunto de test a dataframe para facilitar el trabajo luego
df_X_test = pd.DataFrame(X_test)
df_X_test.columns = var
df_X_test = df_X_test.astype(float)

#Paso el conjunto de train a dataframe para facilitar el trabajo luego
df = pd.DataFrame(X_train)
df.columns = var
df = df.astype(float)

#Matriz de correlación de características
sns.heatmap(df.corr())
plt.title("Correlación de las características (fragmentos intensos)")
plt.show()

#Histograma de las distintas clases
g = sns.displot(data=y_train, kind="hist")
g.set_axis_labels("Clases", "Cantidad de elementos")
g.set_titles("Histograma de Clases")
#g.savefig(fname="Histograma de clases (fragmentos intensos)")
plt.show()

#Muestro los valores faltantes que hay en el train
#print(f"Valores faltantes:{df.isna().sum().sum()}")
    
#Histogramas de las distintas características
pd.DataFrame(df).hist(bins=20,figsize=(15,10))
#plt.savefig(fname="Histogramas (fragmentos intensos)")
plt.show()

#df.info()
#Eliminación de outliers
algunas_var = ['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10','BVP','HR','TEMP','EDA']
z = df[algunas_var].apply(zscore, nan_policy='omit')

df = df[(z<3.5).all(axis=1)]
y_train = y_train[(z<3.5).all(axis=1)]
#print("Despues de eliminar outliers", df.info())

#Standarización de datos
num_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, var)])


##### Support Vector Machine Classifier #####
# svc = SVC(class_weight='balanced',random_state=42, max_iter=1000, kernel='rbf')

svc = SVC(C=16, gamma=0.0625,class_weight='balanced',random_state=42, max_iter=1000, kernel='rbf')
model_svc = Pipeline(steps=[('preprocessor', preprocessor),('svc',svc)])

#Se buscan los mejores parámetros con gridSearch
# param_grid_SVC = [{ "svc__C": [2 ** i for i in range(-2,6)],
#                     "svc__gamma":[2 ** i for i in range(-5,2)]}]

# grid_search_SVC = GridSearchCV(model_svc, param_grid_SVC, cv=5, n_jobs=-1, scoring='balanced_accuracy')
# grid_search_SVC.fit(df,y_train)

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_search_SVC.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_search_SVC.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_search_SVC.best_params_)

'''
Los mejores parámetros han sido:
C=16
gamma=0.0625
'''

scores_svc = cross_val_score(estimator=model_svc, X=df, y=y_train, cv=5, n_jobs=-1, scoring='balanced_accuracy')

print(f"Balanced Accuracy SVC CV 5-Fold (Mean): {np.mean(scores_svc)}")


##### Regresión Logística #####
#lr = LogisticRegression(penalty='l2', n_jobs=-1, class_weight='balanced', random_state=42, max_iter=10000)

lr = LogisticRegression(penalty='l2', n_jobs=-1, class_weight='balanced', random_state=42, max_iter=10000, C=10)
model_lr = Pipeline(steps=[('preprocessor', preprocessor),('lr',lr)])

#Se buscan los mejores parámetros con gridSearch
# param_grid_lr = [{ "lr__C": [0.01,1.0,10,100,1000]}]
# #[0.001,0.01,1.0,10,100,1000,10000,100000]
# grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=5, n_jobs=-1, scoring='balanced_accuracy')
# grid_search_lr.fit(df,y_train)

# print(f"La C escogida ha sido: {grid_search_lr.best_estimator_._final_estimator.C}")
# #La C escogida ha sido:

# # print(" Results from Grid Search " )
# # print("\n The best estimator across ALL searched params:\n",grid_search_lr.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_search_lr.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_search_lr.best_params_)
'''
Los mejores parámetros han sido:
C=10
'''
scores_lr = cross_val_score(estimator=model_lr, X=df, y=y_train, cv=5, n_jobs=-1, scoring='balanced_accuracy')
print(f"Balanced Accuracy Logistic Regression CV 5-Fold (Mean): {np.mean(scores_lr)}")


##### Random Forest #####
#rf = RandomForestClassifier(random_state=42,n_jobs=-1, class_weight='balanced', max_features='sqrt')

rf = RandomForestClassifier(random_state=42,n_jobs=-1, class_weight='balanced', max_features='sqrt', n_estimators=900, max_depth=20)
model_rf = Pipeline(steps=[('preprocessor', preprocessor),('rf',rf)])

#Se buscan los mejores parámetros con gridSearch
# param_grid_rf = [{"rf__n_estimators":[10,20,50,100,200,300,400,500],
#                   "rf__max_depth":[5,10,20,50,100,200]}]

# grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, n_jobs=-1, scoring='balanced_accuracy')
# grid_search_rf.fit(df,y_train)

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_search_rf.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_search_rf.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_search_rf.best_params_)

'''
Los mejores parámetros han sido:
n_estimatorrs = 900
max_depth = 20
'''

scores_rf = cross_val_score(estimator=model_rf, X=df, y=y_train, cv=5, n_jobs=-1, scoring='balanced_accuracy')

print(f"Balanced Accuracy Random Forest CV 5-Fold (Mean): {np.mean(scores_rf)}")


#Random Forest (conjunto de datos fragmentos intensos) 
model_rf.fit(df,y_train)

#Matriz de confusion RF
y_pred = cross_val_predict(model_rf,df,y_train)
confM = confusion_matrix(y_train,y_pred,labels=model_rf.classes_)
confM = np.divide(confM, len(y_train)/100).round(1)
confM = ConfusionMatrixDisplay(confusion_matrix=confM,display_labels=model_rf.classes_)
confM.plot()
plt.title("Matriz de confusión RF (momentos intensos)")
#plt.savefig(fname="Matriz de confusión SVM")
plt.show()


# df_X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
# df_X_test = df_X_test.fillna(df.mean())

# y_pred = model_rf.predict(df_X_test)
# y_pred_train = model_rf.predict(df)


###############################################################################
    #MODELOS PARA EL CONJUNTO DE DATOS REFERENTE A LOS VÍDEOS COMPLETOS#
###############################################################################
print("\n")
print("MODELOS PARA EL CONJUNTO DE DATOS REFERENTE A LOS VÍDEOS COMPLETOS")
print("\n")

#Definción de los plots
sns.set_style("darkgrid")
sns.set_palette("pastel")
sns.despine()

#Obtengo las características y las etiquetas a partir de los datos
datos2 = np.array(datos2)
features2 = datos2[:,:-1]
target2 = datos2[:,-1]

#Separo los datos en test y training
X_train, X_test, y_train, y_test = train_test_split(features2, target2,
               test_size=0.2, random_state=0, shuffle=True, stratify=target2)

#Paso el conjunto de test a dataframe para facilitar el trabajo luego
df_X_test = pd.DataFrame(X_test)
df_X_test.columns = var
df_X_test = df_X_test.astype(float)

#Paso el conjunto de train a dataframe para facilitar el trabajo luego
df = pd.DataFrame(X_train)
df.columns = var
df = df.astype(float)

#Matriz de correlación de características
#Importante! No ejecutar esta matriz de correlación al mismo tiempo que la matriz
#de confusión de los vídeos completos con random forest porque se mezcla. Es decir,
#si se quiere ver la matriz de confusión de vídeos completos hay que dejar esto
#comentado. Y si se quiere ver esto, pues se descomenta y se comenta la matriz de confusión.
# sns.heatmap(df.corr())
# plt.title("Correlación de las características (videos completos)")
# plt.show()

#Histograma de las distintas clases
g = sns.displot(data=y_train, kind="hist")
g.set_axis_labels("Clases", "Cantidad de elementos")
g.set_titles("Histograma de clases (videos completos)")
#g.savefig(fname="Histograma de clases (videos completos)")
plt.show()

#Muestro los valores faltantes que hay en el train
#print(f"Valores faltantes:{df.isna().sum().sum()}")
    
#Histogramas de las distintas características
pd.DataFrame(df).hist(bins=20,figsize=(15,10))
#plt.savefig(fname="Histogramas (videos completos)")
plt.show()

#df.info()
#Eliminación de outliers
z = df[algunas_var].apply(zscore, nan_policy='omit')

df = df[(z<3.5).all(axis=1)]
y_train = y_train[(z<3.5).all(axis=1)]
#print("Despues de eliminar outliers-----------------------------------------")
#print(df.info())

#Standarización de datos
num_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, var)])


##### Support Vector Machine Classifier #####
#svc = SVC(class_weight='balanced',random_state=42, max_iter=1000, kernel='rbf')

svc = SVC(C=16, gamma=0.125,class_weight='balanced',random_state=42, max_iter=1000, kernel='rbf')
model_svc = Pipeline(steps=[('preprocessor', preprocessor),('svc',svc)])

#Se buscan los mejores parámetros con gridSearch
# param_grid_SVC = [{ "svc__C": [2 ** i for i in range(-2,6)],
#                     "svc__gamma":[2 ** i for i in range(-5,2)]}]

# grid_search_SVC = GridSearchCV(model_svc, param_grid_SVC, cv=5, n_jobs=-1, scoring='balanced_accuracy')
# grid_search_SVC.fit(df,y_train)

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_search_SVC.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_search_SVC.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_search_SVC.best_params_)

'''
Los mejores parámetros han sido:
C=16
gamma=0.125
'''

scores_svc = cross_val_score(estimator=model_svc, X=df, y=y_train, cv=5, n_jobs=-1, scoring='balanced_accuracy')

print(f"Balanced Accuracy SVC CV 5-Fold (Mean): {np.mean(scores_svc)}")


##### Regresión Logística #####
#lr = LogisticRegression(penalty='l2', n_jobs=-1, class_weight='balanced', random_state=42, max_iter=10000)

lr = LogisticRegression(penalty='l2', n_jobs=-1, class_weight='balanced', random_state=42, max_iter=10000, C=1)
model_lr = Pipeline(steps=[('preprocessor', preprocessor),('lr',lr)])

# param_grid_lr = [{ "lr__C": [0.001,0.01,1.0,10,100,1000,10000]}]

#Se buscan los mejores parámetros con gridSearch
# grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=5, n_jobs=-1, scoring='balanced_accuracy')
# grid_search_lr.fit(df,y_train)

# print(f"La C escogida ha sido: {grid_search_lr.best_estimator_._final_estimator.C}")

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_search_lr.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_search_lr.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_search_lr.best_params_)

# lr = LogisticRegression(penalty='l2', n_jobs=-1, class_weight='balanced', random_state=42, max_iter=10000, C=10)

'''
Los mejores parámetros han sido:
C=1
'''

scores_lr = cross_val_score(estimator=model_lr, X=df, y=y_train, cv=5, n_jobs=-1, scoring='balanced_accuracy')

print(f"Balanced Accuracy Logistic Regression CV 5-Fold (Mean): {np.mean(scores_lr)}")


##### Random Forest #####
#rf = RandomForestClassifier(random_state=42,n_jobs=-1, class_weight='balanced', max_features='sqrt')

rf = RandomForestClassifier(random_state=42,n_jobs=-1, class_weight='balanced', max_features='sqrt', n_estimators=500, max_depth=20)
model_rf = Pipeline(steps=[('preprocessor', preprocessor),('rf',rf)])

#Se buscan los mejores parámetros con gridSearch
# param_grid_rf = [{"rf__n_estimators":[50,100,200,300,400,500,600],
#                   "rf__max_depth":[10,20,50,100,200]}]

# grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, n_jobs=-1, scoring='balanced_accuracy')
# grid_search_rf.fit(df,y_train)

# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_search_rf.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_search_rf.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_search_rf.best_params_)

'''
Los mejores parámetros han sido:
n_estimatorrs = 500
max_depth = 20
'''

scores_rf = cross_val_score(estimator=model_rf, X=df, y=y_train, cv=5, n_jobs=-1, scoring='balanced_accuracy')

print(f"Balanced Accuracy Random Forest CV 5-Fold (Mean): {np.mean(scores_rf)}")


#Modelo escogido Random Forest (conjunto de datos vídeos completos)
model_rf.fit(df,y_train)

#Matriz de confusion RF
y_pred = cross_val_predict(model_rf,df,y_train)
confM = confusion_matrix(y_train,y_pred,labels=model_rf.classes_)
confM = np.divide(confM, len(y_train)/100).round(1)
confM = ConfusionMatrixDisplay(confusion_matrix=confM, display_labels=model_rf.classes_)
confM.plot()
plt.title("Matriz de confusión RF (videos completos)")
#plt.savefig(fname="Matriz de confusión SVM")
plt.show()

df_X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_X_test = df_X_test.fillna(df.mean())

y_pred = model_rf.predict(df_X_test)
y_pred_train = model_rf.predict(df)

print(f"Balanced Accuracy (test) para  Random Forest (entrenando con todo el training): {balanced_accuracy_score(y_true=y_test,y_pred=y_pred)}")

