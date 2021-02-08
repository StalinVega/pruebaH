from flask import Flask, render_template
import pymysql.cursors


app = Flask(__name__)
#coneccion
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='proyecto',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)



with connection.cursor() as cursor:
    sql = "SELECT tema,keywords,contenedor,categoria FROM base "
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result[0])