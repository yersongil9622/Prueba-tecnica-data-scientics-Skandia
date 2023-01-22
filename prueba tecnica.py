# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 08:52:36 2023

@author: yerso
"""

"---------------------------------- Carga de librerias  ------------------------"

import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import re
"----------------------------------- Carga de datos ---------------------------"

Transferencias = pd.read_parquet('C:/Users/yerso/OneDrive/0transferencias.parquet')
Saldos = pd.read_parquet('C:/Users/yerso/OneDrive/0saldos.parquet')
Clientes = pd.read_parquet('C:/Users/yerso/OneDrive/0clientes.parquet')

"------------------------ Tratamiento de datos trasferencias -------------------------------"

Transferencias=Transferencias[Transferencias["FechaEfectiva"]>='2021-01-01']

# Creo variables adicionales que ayudaran al modelo #

" en mi experiencia variables de aportes y retiros ayudan a visualizar o comprender mejor los patrones de descapitalizacion \
    que pueda tener un cliente o de plano mostrar una fuerte tendencia a dejar en 0 las cuentas\
        adicionalmente estandarizo una variable para tener una fecha corte para todas las transacciones hehcas en un mes\
            especifico de tal manera que pueda organizar toda la informacion en meses"
    
Transferencias["mes_corte"]=Transferencias["FechaEfectiva"].apply(lambda x: x.to_period('M').to_timestamp('D', how='end'))
Transferencias["num_aportes"]=np.where(Transferencias["ValorNeto"]>=0,1,0)
Transferencias["num_retiros"]=np.where(Transferencias["ValorNeto"]<0,1,0)
Transferencias["val_aportes"]=np.where(Transferencias["ValorNeto"]>=0,Transferencias["ValorNeto"],0)
Transferencias["val_retiros"]=np.where(Transferencias["ValorNeto"]<0,Transferencias["ValorNeto"],0)

"Aca agrupo todo con el fin de dejar una dataset en meses, adicionalmente se agruppa por contrato y plan"

total_traf=Transferencias.groupby(["Contrato","PlanProducto","mes_corte"],as_index=False).agg(
    {
     "num_aportes":"sum",
     "num_retiros":"sum",
     "val_aportes":"sum",
     "val_retiros":"sum"
     }
    )

total_traf['mes_corte']=total_traf['mes_corte'].apply(lambda x: x.strftime('%Y-%m-%d'))

" aca pivoteo la informacion de tal manera que cada  contrato producto quede con un registro\
    por columna de sus actividades mes a mes,algo muy siimilar a la talba de saldos"
    
total_traf=total_traf.pivot_table(index=['Contrato',"PlanProducto"],columns='mes_corte',aggfunc="sum")
total_traf=total_traf.fillna(0)
total_traf.columns = list(map("_".join, total_traf.columns))
total_traf=total_traf.reset_index()

"------------------------ Tratamiento de datos Saldos -------------------------------"

"en esta parte se empieza a hacer el tratamiento de la base de datos de los Saldos, primero que todoeliminare \
    eliminare contratos que en todo el periodo de estudio no hayan tenido saldos en ningun mes para eso \
        hago una suma de tosos los meses y los que sean iguales a 0 seran eliminados"
        
 
Saldos["sum_saldos"]=Saldos[[ 'SALDO_202101',
 'SALDO_2021O2',
 'SALDO_2021O3',
 'SALDO_2021O4',
 'SALDO_2021O5',
 'SALDO_2021O6',
 'SALDO_2021O7',
 'SALDO_2021O8',
 'SALDO_2021O9',
 'SALDO_202110',
 'SALDO_202111',
 'SALDO_202112',
 'SALDO_202201',
 'SALDO_202202',
 'SALDO_202203',
 'SALDO_202204',
 'SALDO_202205',
 'SALDO_202206',
 'SALDO_202207',
 'SALDO_202208',
 'SALDO_202209',
 "SALDO_202210"]].sum(axis=1)

Saldos=Saldos[Saldos["sum_saldos"]!=0]

print(Saldos.SALDO_202209.sum()) # identifico que los Saldos de septiembre en todos los clientes son 0 (¿porque?)
                             # por tanto no se tomara ese mes para ningun tipo de analisis, por tanto sale de los dataset de saldos

Saldos=Saldos[[
 'SALDO_202101',
 'SALDO_2021O2',
 'SALDO_2021O3',
 'SALDO_2021O4',
 'SALDO_2021O5',
 'SALDO_2021O6',
 'SALDO_2021O7',
 'SALDO_2021O8',
 'SALDO_2021O9',
 'SALDO_202110',
 'SALDO_202111',
 'SALDO_202112',
 'SALDO_202201',
 'SALDO_202202',
 'SALDO_202203',
 'SALDO_202204',
 'SALDO_202205',
 'SALDO_202206',
 'SALDO_202207',
 'SALDO_202208',
 "SALDO_202210",
 'Contrato',
 'PlanProducto',
 'NroDocum']]


"----------------------------- Join con clientes --------------------"

"organizada la data de los saldos y resumida las trasferncias en meses se puede\
    hacer un join entre las tres bases con el fin de crear una base unica por cliente resumiendo todos los\
        saldos y trasacciones de los clientes en los 22 meses"


Clientes=Clientes[["NroDocum","TIPODOCUM","CIUDAD"]]

Clien_prod=Clientes.merge(Saldos,how="left",on="NroDocum")

Clien_prod=Clien_prod.merge(total_traf,how="left",on=['Contrato','PlanProducto'])

list(Clien_prod)

"------------------------------------- Agrupacion por cliente -------------------------------"
"Aca en adelante se hace agrupacion por el cliente, sumaré saldos y transacciones\
    adiciolnalmete tendre en cuenta los productos y planes que tenga el cliente "
    
base = Clien_prod.groupby("NroDocum",as_index=False).agg(
  {
    'Contrato':"count",
    'PlanProducto':"count",
    'SALDO_202101':"sum",
    'SALDO_2021O2':"sum",
    'SALDO_2021O3':"sum",
    'SALDO_2021O4':"sum",
    'SALDO_2021O5':"sum",
    'SALDO_2021O6':"sum",
    'SALDO_2021O7':"sum",
    'SALDO_2021O8':"sum",
    'SALDO_2021O9':"sum",
    'SALDO_202110':"sum",
    'SALDO_202111':"sum",
    'SALDO_202112':"sum",
    'SALDO_202201':"sum",
    'SALDO_202202':"sum",
    'SALDO_202203':"sum",
    'SALDO_202204':"sum",
    'SALDO_202205':"sum",
    'SALDO_202206':"sum",
    'SALDO_202207':"sum",
    'SALDO_202208':"sum",
    "SALDO_202210":"sum",
    'num_aportes_2021-01-31':"sum",
    'num_aportes_2021-02-28':"sum",
    'num_aportes_2021-03-31':"sum",
    'num_aportes_2021-04-30':"sum",
    'num_aportes_2021-05-31':"sum",
    'num_aportes_2021-06-30':"sum",
    'num_aportes_2021-07-31':"sum",
    'num_aportes_2021-08-31':"sum",
    'num_aportes_2021-09-30':"sum",
    'num_aportes_2021-10-31':"sum",
    'num_aportes_2021-11-30':"sum",
    'num_aportes_2021-12-31':"sum",
    'num_aportes_2022-01-31':"sum",
    'num_aportes_2022-02-28':"sum",
    'num_aportes_2022-03-31':"sum",
    'num_aportes_2022-04-30':"sum",
    'num_aportes_2022-05-31':"sum",
    'num_aportes_2022-06-30':"sum",
    'num_aportes_2022-07-31':"sum",
    'num_aportes_2022-08-31':"sum",
    'num_aportes_2022-09-30':"sum",
    'num_aportes_2022-10-31':"sum",
    'num_retiros_2021-01-31':"sum",
    'num_retiros_2021-02-28':"sum",
    'num_retiros_2021-03-31':"sum",
    'num_retiros_2021-04-30':"sum",
    'num_retiros_2021-05-31':"sum",
    'num_retiros_2021-06-30':"sum",
    'num_retiros_2021-07-31':"sum",
    'num_retiros_2021-08-31':"sum",
    'num_retiros_2021-09-30':"sum",
    'num_retiros_2021-10-31':"sum",
    'num_retiros_2021-11-30':"sum",
    'num_retiros_2021-12-31':"sum",
    'num_retiros_2022-01-31':"sum",
    'num_retiros_2022-02-28':"sum",
    'num_retiros_2022-03-31':"sum",
    'num_retiros_2022-04-30':"sum",
    'num_retiros_2022-05-31':"sum",
    'num_retiros_2022-06-30':"sum",
    'num_retiros_2022-07-31':"sum",
    'num_retiros_2022-08-31':"sum",
    'num_retiros_2022-09-30':"sum",
    'num_retiros_2022-10-31':"sum",
    'val_aportes_2021-01-31':"sum",
    'val_aportes_2021-02-28':"sum",
    'val_aportes_2021-03-31':"sum",
    'val_aportes_2021-04-30':"sum",
    'val_aportes_2021-05-31':"sum",
    'val_aportes_2021-06-30':"sum",
    'val_aportes_2021-07-31':"sum",
    'val_aportes_2021-08-31':"sum",
    'val_aportes_2021-09-30':"sum",
    'val_aportes_2021-10-31':"sum",
    'val_aportes_2021-11-30':"sum",
    'val_aportes_2021-12-31':"sum",
    'val_aportes_2022-01-31':"sum",
    'val_aportes_2022-02-28':"sum",
    'val_aportes_2022-03-31':"sum",
    'val_aportes_2022-04-30':"sum",
    'val_aportes_2022-05-31':"sum",
    'val_aportes_2022-06-30':"sum",
    'val_aportes_2022-07-31':"sum",
    'val_aportes_2022-08-31':"sum",
    'val_aportes_2022-09-30':"sum",
    'val_aportes_2022-10-31':"sum",
    'val_retiros_2021-01-31':"sum",
    'val_retiros_2021-02-28':"sum",
    'val_retiros_2021-03-31':"sum",
    'val_retiros_2021-04-30':"sum",
    'val_retiros_2021-05-31':"sum",
    'val_retiros_2021-06-30':"sum",
    'val_retiros_2021-07-31':"sum",
    'val_retiros_2021-08-31':"sum",
    'val_retiros_2021-09-30':"sum",
    'val_retiros_2021-10-31':"sum",
    'val_retiros_2021-11-30':"sum",
    'val_retiros_2021-12-31':"sum",
    'val_retiros_2022-01-31':"sum",
    'val_retiros_2022-02-28':"sum",
    'val_retiros_2022-03-31':"sum",
    'val_retiros_2022-04-30':"sum",
    'val_retiros_2022-05-31':"sum",
    'val_retiros_2022-06-30':"sum",
    'val_retiros_2022-07-31':"sum",
    'val_retiros_2022-08-31':"sum",
    'val_retiros_2022-09-30':"sum",
    'val_retiros_2022-10-31':"sum"
  })

"Revisando la base resumida por cliente se encuentra que muchos clientes no tienen productos activos o por lo menos con\
    saldos en algun momento del periodo de estudio, por ende se eliminaran de el dataset para tener infomarcion de mas calidad"

base=base[base["Contrato"]>0]

"------------------------------------- Creacion de la variable dependiente o de respuesta ----------------------------"

"para crear la variable de respuesta y segun la solicitud usare el saldo mas reciente y el saldo de tres meses atras\
    la formula para extraer el indice de descapitalizacion sera (t2/t1)-1, de tal manera que si el indice es menor a -0.7\
        sera marcado como un cliente descapitalizado de lo contratio no sera marcado"

base["capitalcion"]=np.nan

"ahora bien, en la revision de la base de datos muestra que algunos clientes retiraron todo su capital antes de octubre del 2022\
    y en diferntes meses, por tanto para recuperar estas marcaciones y crear el indice de forma mas exacta"

base["mes_retiro"]=np.nan

Variables= ['SALDO_202101','SALDO_2021O2','SALDO_2021O3','SALDO_2021O4',
          'SALDO_2021O5','SALDO_2021O6','SALDO_2021O7','SALDO_2021O8',
          'SALDO_2021O9','SALDO_202110','SALDO_202111','SALDO_202112',
          'SALDO_202201','SALDO_202202','SALDO_202203','SALDO_202204',
          'SALDO_202205','SALDO_202206','SALDO_202207','SALDO_202208',
          "SALDO_202210"]

Variables.reverse()

for i in range(len(base)):
    for j in Variables:
        if base[j].iloc[i]!=0:
            break
        base["mes_retiro"].iloc[i]=j
        try:
            base["capitalcion"].iloc[i]=((base[j].iloc[i])/base[Variables[Variables.index(j)+2]].iloc[i])-1
        except:
            base["capitalcion"].iloc[i]=((base[j].iloc[i])/base[Variables[Variables.index(j)+1]].iloc[i])-1

base["capitalcion"]=np.where(base["capitalcion"].isna(),(base["SALDO_202210"]/base["SALDO_202208"])-1,
                               base["capitalcion"])

base["target"]=np.where(base["capitalcion"]<=-0.7,1,0)

"reviso el balnceo de la data, este esta un 85-15 mas o menos, lo que podria ser un problema para los modelos,\
se podria hacer una revision de tecnicas de balanceo pero se hara el ejercicio de modelado con ese dataset"

print(base["target"].value_counts()/base["target"].value_counts().sum()) 

"Aca se alistan las variables que usare para el modelo, en un ejercicio mas amplio se debe hacer una analisis univariado\
    multivariado y ACP(análisis de componentes principales) para hacer depuracion de variables \
        de tal manera se entrega un modelo lo mas limpio y eficiente."


varX=['Contrato',
 'PlanProducto',
 'SALDO_202101',
 'SALDO_2021O2',
 'SALDO_2021O3',
 'SALDO_2021O4',
 'SALDO_2021O5',
 'SALDO_2021O6',
 'SALDO_2021O7',
 'SALDO_2021O8',
 'SALDO_2021O9',
 'SALDO_202110',
 'SALDO_202111',
 'SALDO_202112',
 'SALDO_202201',
 'SALDO_202202',
 'SALDO_202203',
 'SALDO_202204',
 'SALDO_202205',
 'SALDO_202206',
 'SALDO_202207',
 'num_aportes_2021-01-31',
 'num_aportes_2021-02-28',
 'num_aportes_2021-03-31',
 'num_aportes_2021-04-30',
 'num_aportes_2021-05-31',
 'num_aportes_2021-06-30',
 'num_aportes_2021-07-31',
 'num_aportes_2021-08-31',
 'num_aportes_2021-09-30',
 'num_aportes_2021-10-31',
 'num_aportes_2021-11-30',
 'num_aportes_2021-12-31',
 'num_aportes_2022-01-31',
 'num_aportes_2022-02-28',
 'num_aportes_2022-03-31',
 'num_aportes_2022-04-30',
 'num_aportes_2022-05-31',
 'num_aportes_2022-06-30',
 'num_aportes_2022-07-31',
 'num_retiros_2021-01-31',
 'num_retiros_2021-02-28',
 'num_retiros_2021-03-31',
 'num_retiros_2021-04-30',
 'num_retiros_2021-05-31',
 'num_retiros_2021-06-30',
 'num_retiros_2021-07-31',
 'num_retiros_2021-08-31',
 'num_retiros_2021-09-30',
 'num_retiros_2021-10-31',
 'num_retiros_2021-11-30',
 'num_retiros_2021-12-31',
 'num_retiros_2022-01-31',
 'num_retiros_2022-02-28',
 'num_retiros_2022-03-31',
 'num_retiros_2022-04-30',
 'num_retiros_2022-05-31',
 'num_retiros_2022-06-30',
 'num_retiros_2022-07-31',
 'val_aportes_2021-01-31',
 'val_aportes_2021-02-28',
 'val_aportes_2021-03-31',
 'val_aportes_2021-04-30',
 'val_aportes_2021-05-31',
 'val_aportes_2021-06-30',
 'val_aportes_2021-07-31',
 'val_aportes_2021-08-31',
 'val_aportes_2021-09-30',
 'val_aportes_2021-10-31',
 'val_aportes_2021-11-30',
 'val_aportes_2021-12-31',
 'val_aportes_2022-01-31',
 'val_aportes_2022-02-28',
 'val_aportes_2022-03-31',
 'val_aportes_2022-04-30',
 'val_aportes_2022-05-31',
 'val_aportes_2022-06-30',
 'val_aportes_2022-07-31',
 'val_retiros_2021-01-31',
 'val_retiros_2021-02-28',
 'val_retiros_2021-03-31',
 'val_retiros_2021-04-30',
 'val_retiros_2021-05-31',
 'val_retiros_2021-06-30',
 'val_retiros_2021-07-31',
 'val_retiros_2021-08-31',
 'val_retiros_2021-09-30',
 'val_retiros_2021-10-31',
 'val_retiros_2021-11-30',
 'val_retiros_2021-12-31',
 'val_retiros_2022-01-31',
 'val_retiros_2022-02-28',
 'val_retiros_2022-03-31',
 'val_retiros_2022-04-30',
 'val_retiros_2022-05-31',
 'val_retiros_2022-06-30',
 'val_retiros_2022-07-31']

varY=["target"]
"---------------------------- Modelado ---------------------------------------"


"De aca en adelante empiezo la etapa de modelado, probare las metodologias mas conocidas\
    en la ciencia de datos, las cuales son: Regresiones logisticas, KNN, \
    maquinas de soporte vectorial, NB gaussian,arboles de decision y \
    bosques aleatorios. se penso en usar metodologias un poco mas robustas como:\
    XGboots,LightGBM,Catboots y redes neuronales, sin embargo los resultados con \
    las primeros algortimos mencionados fueron bastante buenos."

X=base[varX]
Y=base[varY]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve

# Creo los set de entrenamiento y validacion,un 70 entrenamiento un 30 de testeo.
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=474) 

# ---------------- Regresion logistica ------------------------------- #


LRclassifier = LogisticRegression(class_weight="balanced")
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))

# ---------------- KNN --------------------------------------- #


KNclassifier = KNeighborsClassifier(n_neighbors=5)
KNclassifier.fit(X_train, y_train)

y_pred = KNclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

KNAcc = accuracy_score(y_pred,y_test)
print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc*100))

# --------------------- Maquina de soporte vectorial -------------------- #

SVCclassifier = SVC()
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc*100))

#---------------------- NB gaussian ----------------------------------- #


NBclassifier2 = GaussianNB()
NBclassifier2.fit(X_train, y_train)

y_pred = NBclassifier2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


NBAcc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy is: {:.2f}%'.format(NBAcc2*100))

#-------------------- Arbol de decisiones ---------------------------- #

DTclassifier = DecisionTreeClassifier(max_depth=10,
                                      random_state=123,
                                      class_weight="balanced")
DTclassifier.fit(X_train, y_train)

y_pred = DTclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

importance=DTclassifier.feature_importances_
dfplot=pd.DataFrame()
dfplot["colnames"]=varX
dfplot["importancia"]=importance
dfplot=dfplot[dfplot["importancia"]>0]
dfplot=dfplot.sort_values("importancia",ascending=False)
sns.set(rc={'figure.figsize':(30,30)})
g=sns.barplot(x="importancia", y = "colnames" , data = dfplot)

DTAcc = accuracy_score(y_pred,y_test)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc*100))

# --------------------- Bosque aleatorio -------------------------- #

RFclassifier = RandomForestClassifier(max_depth=10,random_state=123,class_weight="balanced")
RFclassifier.fit(X_train, y_train)

y_pred = RFclassifier.predict(X_test)
y_proba = RFclassifier.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


importance=RFclassifier.feature_importances_
dfplot=pd.DataFrame()
dfplot["colnames"]=varX
dfplot["importancia"]=importance
dfplot=dfplot[dfplot["importancia"]>0]
dfplot=dfplot.sort_values("importancia",ascending=False)
sns.set(rc={'figure.figsize':(30,30)})
g=sns.barplot(x="importancia", y = "colnames" , data = dfplot)

RFAcc = accuracy_score(y_pred,y_test)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))


"------------------------------- Compracacion de modelos --------------------"

"ya como ultima parte hago una comparacion del Accuracy de todos los modelos con el fin de ver cuales entregan mejores resultados"

compare = pd.DataFrame({'Model': ['Logistic Regression', 'K Neighbors', 'SVM', 'Gaussian NB', 'Decision Tree', 'Random Forest'], 
                        'Accuracy': [LRAcc*100, KNAcc*100, SVCAcc*100, NBAcc2*100, DTAcc*100,  RFAcc*100]})
print(compare.sort_values(by='Accuracy', ascending=False))

"con estos resultados concluyo que el mejor modelo son los bosques aleatorios, al revisar su matrx de confuccion y\
    el reporte de clasificacion este es el que mas minimiza los errores tipo I y tipo II,\
    sin embargo no descarteria del todo el arbol de decision ya que sus resultados no son nada despreciables adicionalmente\
    el poder explicativo de el arbol de decision es bastante importante a la hora de auditorias(muy comunes en el sistema financiero)\
    ahora bien la depuracion de las variables es de suma importancia, por tanto hay que hacerla en este ejecicio, por eso creo\
    una grafica para encontrar aquellas que poco o nada estan aportando al modelo y asi en un trabajo mas arduo irlas sacando del modelo"
    
"---------------------------- optimizacion basica del modelo -----------------------"

"como un ejercicio adicional, se puede hacer una optimizacion de los parametros del modelo d forma individual o multiple\
    en este caso solo hare una optimizacion basica delbosque aleatorio usando un for pero si s quiere se podria hacer una optimizacion\
        dinamica usando metodos mas robustos"
        
scoreListRF = []
parameter = []
for i in range(2,40):
    RFclassifier = RandomForestClassifier(max_depth=i,random_state=123,
                                          class_weight="balanced")
    RFclassifier.fit(X_train, y_train)
    parameter.append(i)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    
plt.plot(range(2,40), scoreListRF)
plt.xticks(np.arange(2,40,5))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAccMax = max(scoreListRF)
parametros=pd.DataFrame()
parametros["parametro"]=parameter
parametros["resultado"]=scoreListRF
print("RF Acc Max {:.2f}%".format(RFAccMax*100))

# --------------------- Bosque aleatorio (optimizado) -------------------------- #

RFclassifier_op = RandomForestClassifier(max_depth=9,
                                      random_state=123
                                      ,class_weight="balanced")
RFclassifier_op.fit(X_train, y_train)

y_pred_op = RFclassifier_op.predict(X_test)
y_proba_op = RFclassifier_op.predict_proba(X_test)[:,1]

RFAcc = accuracy_score(y_pred_op,y_test)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))

print("------------------- Modelo Basico---------------------------")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("------------------- Modelo optimizado---------------------------")

print(classification_report(y_test, y_pred_op))
print(confusion_matrix(y_test, y_pred_op))

"la optimizacion mejora un poco los resultados, sim embargo esta puede hacerse a varios parametros y norrmalmente\
    se hace en varios parametros en conjunto"