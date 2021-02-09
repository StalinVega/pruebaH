from flask import Flask, render_template
import pymysql.cursors
import re
from unicodedata import normalize
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/')
def index():
    nltk.download('stopwords')

    sw = pd.read_excel("software.xlsx", engine='openpyxl')
    tits = np.array(sw['tema'])
    keyws = np.array(sw['keywords'])
    tems = np.array(sw['contenedor'])

    red = pd.read_excel("redes.xlsx")
    titr = np.array(red['tema'])
    keywr = np.array(red['keywords'])
    temr = np.array(red['contenedor'])

    mach = pd.read_excel("machine.xlsx")
    titm = np.array(mach['tema'])
    keywm = np.array(mach['keywords'])
    temm = np.array(mach['contenedor'])

    base = pd.read_excel("base.xlsx")
    titb = np.array(base['tema'])
    keywb = np.array(base['keywords'])
    temb = np.array(base['contenedor'])

    general = pd.read_excel("Libro2.xlsx")
    titge = np.array(general['tema'])
    keywge = np.array(general['keywords'])
    temge = np.array(general['contenedor'])
    y = np.array(general['categoria'])

    # NPL
    ###############################################################################
    def documentNPL(lists=[]):
        charDoc = []
        temp = ""
        for count, doc in enumerate(lists):
            temp = str(doc)
            trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
            temp = normalize('NFKC', normalize('NFKD', temp).translate(trans_tab))
            temp = re.sub('[^A-Za-z0-9]+', ' ', temp.lower())
            charDoc.append(temp)
        stopw = stopwords.words('spanish')
        # ps = PorterStemmer()
        stopDoc = []
        for i in charDoc:
            line = ""
            for j in i.split():
                if not j in stopw:
                    # temp = ps.stem(j)
                    line = line + j + " "
            stopDoc.append(line)
        return stopDoc

    def tokenDic(list=[]):
        dicc = []
        for i in list:
            # for j in i.split():
            if not i in dicc:
                dicc.append(i)
        return dicc

    # Software
    NPL_keys = documentNPL(keyws)
    NPL_conts = documentNPL(tems)
    NPL_titls = documentNPL(tits)

    alls = NPL_keys + NPL_conts + NPL_titls
    diccs = tokenDic(alls)

    bg_software = ""
    for i in range(0, len(diccs)):
        bg_software += diccs[i]

    # Machine Learning
    NPL_keym = documentNPL(keywm)
    NPL_contm = documentNPL(temm)
    NPL_titlm = documentNPL(titm)

    allm = NPL_keym + NPL_contm + NPL_titlm
    diccm = tokenDic(allm)

    bg_ml = ""
    for i in range(0, len(diccm)):
        bg_ml += diccm[i]

    # Redes
    NPL_keyr = documentNPL(keywr)
    NPL_contr = documentNPL(temr)
    NPL_titlr = documentNPL(titr)

    allr = NPL_keyr + NPL_contm + NPL_titlr
    diccr = tokenDic(allr)

    bg_redes = ""
    for i in range(0, len(diccr)):
        bg_redes += diccr[i]

    # base
    NPL_keyb = documentNPL(keywb)
    NPL_contb = documentNPL(temb)
    NPL_titlb = documentNPL(titb)

    allb = NPL_keyb + NPL_contb + NPL_titlb
    diccb = tokenDic(allb)

    bg_base = ""
    for i in range(0, len(diccb)):
        bg_base += diccb[i]

    # --------------------------------------------------------------------------------------

    def jaccard(a, b):
        str1 = set(a.split())
        str2 = set(b.split())
        return float(len(str1 & str2)) / len(str1 | str2)

    def similitud_con_bg(l, ll):
        a1 = l
        b1 = ll
        d = []
        for i in range(0, len(a1)):
            d.append(jaccard(str(a1[i]), b1))
        return d

    t_s = similitud_con_bg(titge, bg_software)
    t_ml = similitud_con_bg(titge, bg_ml)
    t_redes = similitud_con_bg(titge, bg_redes)
    t_base = similitud_con_bg(titge, bg_base)

    k_s = similitud_con_bg(keywge, bg_software)
    k_ml = similitud_con_bg(keywge, bg_ml)
    k_redes = similitud_con_bg(keywge, bg_redes)
    k_base = similitud_con_bg(keywge, bg_base)

    a_s = similitud_con_bg(temge, bg_software)
    a_ml = similitud_con_bg(temge, bg_ml)
    a_redes = similitud_con_bg(temge, bg_redes)
    a_base = similitud_con_bg(temge, bg_base)

    X = list(zip(t_s, t_ml, t_redes, t_base, k_s, k_ml, k_redes, k_base, a_s, a_ml, a_redes, a_base))

    ############################################################
    print(
        '---------------------------------------Modelo de Regresión Logística--------------------------------------------')

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,random_state=0)  # Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos con un entrenamiento del 70% y para el test 30%

    ##Algoritmo de regresion logistica
    algoritmo = LogisticRegression(solver='liblinear', multi_class='ovr')
    algoritmo.fit(X_train, y_train)
    Y_pred = algoritmo.predict(X_test)
    press = algoritmo.score(X_test, y_test)
    print('Precisión Regresión Logística: {}'.format(algoritmo.score(X_test, y_test)))

    # Calculo la exactitud del modelo
    from sklearn.metrics import accuracy_score
    exactitud = accuracy_score(y_test, Y_pred, )

    cm = confusion_matrix(y_test, Y_pred)

    def recall(label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def recall_macro_average(confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += recall(label, confusion_matrix)
        return sum_of_recalls / columns

    print("Recall total:", recall_macro_average(cm))
    reL = recall_macro_average(cm)

    mm = classification_report(y_test, Y_pred)
    #######################KNN###################################3
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # colocamos el valor de K en 7
    n_neighbors = 7

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train, y_train)))
    acuTr = knn.score(X_train, y_train)
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test, y_test)))
    acuTest = knn.score(X_test, y_test)

    # Matriz de confusion

    pred = knn.predict(X_test)
    cm1 = confusion_matrix(y_test, pred)
    print(classification_report(y_test, pred))

    ###########################################

    def precision(label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def precision_macro_average(confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            sum_of_precisions += precision(label, confusion_matrix)
        return sum_of_precisions / rows

    def recall_macro_average(confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += recall(label, confusion_matrix)
        return sum_of_recalls / columns

    print("Precision total:", precision_macro_average(cm1))
    print("RECALL total:", recall_macro_average(cm1))
    rc = recall_macro_average(cm1)
    #############################################
    # ELEGIR EL MEJOR VALOR DE K

    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.savefig("templates/image/acurracy-knn-vecinos-k.png", depi=100)

    return render_template('index.html', exactitud=exactitud, press=press, mm=mm, acuTr=acuTr, acuTest=acuTest, rc=rc,
                           reL=reL, pred=pred[-1], Y_pred=Y_pred[-1])


if __name__ == '__main__':
    app.run()

