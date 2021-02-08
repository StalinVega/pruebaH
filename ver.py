# CODIGO DE EXTRACION DE LA BASE DE DATOS EL DATASET
def mcx():
    sql = "SELECT CANTIDAD,LARGO,ANCHO,ALTO,ESTADO,PRECIO FROM MC"
    cursor = db.cursor()
    cursor.execute(sql)
    pedct = cursor.fetchall()
    x = []
    y = []
    mat = []
    maty = []
    for row in pedct:
        mat = [row[0], row[1], row[2], row[3], row[4]]
        maty = [row[5]]
        y.append((maty))
        x.append(mat)
    return x, y


# CODIGO DONDE SE CALCULA LO DE MACHINE LEARNING

@app.route('/reportes')
def reportes() -> 'html':
    import pandas as pd
    if 'username' in session:
        menu = menuopciones(str(session['perfiluser']))
        usuario = session['username']
        nom = str(session['nombre'])
        ci = str(session['ci'])
        ced = session['ci']

        # LLAMADO AL METODO DE LA BASE DE DATOS
        mat, maty = mcx()

        # SE TRANSFORMA MAT EN UN DATAFRAME VALORES DE (X) PARA LA REGRESION
        a = np.array(pd.DataFrame(mat))

        # VALORES DE Y PARA LA REGRESION
        b = np.array(maty)

        # DIVISION DEL DATASET EN TRAINING Y TEST
        X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.3, random_state=40)

        # CALCULO DE LA REGRESION LINEAL
        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        lr.coef_
        lr.intercept_

        # VECTOR DE PREDICCIONES
        Y_pred = lr.predict(X_test)
        # print(Y_pred)

        # PRESICION DEL ALGORITMO
        precision = lr.score(X_train, y_train)
        algo = "ALGORITMO DE REGRESION LINEAL MULTIPLE"

        def media(valores):
            return sum(valores) / len(valores)

        def evaluacion_rendimiento(yt, ypre):
            error = yt - ypre
            print(error)
            MAE = sum(abs(error)) / len(error)
            MSE = sum(pow(error, 2)) / len(error)
            RMSE = math.sqrt(MSE)
            SCE = sum(pow(error, 2))
            median = float(media(yt))
            STC = sum(pow(yt - median, 2))
            SCR = STC - SCE
            c = str(lr.coef_)
            coef = c.split()
            r2 = SCR / STC
            r2_adj = 1 - (1 - r2) * ((len(y_test) - 1) / (len(yt) - (len(coef) - 1) - 1))

            return MAE, MSE, RMSE, r2, r2_adj

        # EVALUACION DE RENDIMIENTO
        MAE1, MSE1, RMSE1, r21, r2_adj1 = evaluacion_rendimiento(y_test, Y_pred)
        ma = round(float(MAE1), 2)
        mse = round(float(MSE1), 2)
        rm = round(float(RMSE1), 2)
        rs = round(float(r21), 2)
        r2a = round(float(r2_adj1), 2)

        # ENVIO DE LOS DATOS HACIA LA PAGINA WEB

        return render_template('reportes.html', menu=menu, user=usuario, nombreusuario=nom, codigousuario=ci,
                               pre=precision, al=algo,
                               mae=ma, msee=mse, rmse=rm, r2=rs, r2aa=r2a)
    else:
        return redirect(url_for('index'))