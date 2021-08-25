# from typing import Any
from flask import Flask, request, session,g,redirect,url_for, abort, flash, render_template, jsonify, json
import pandas as pd

import matplotlib.pyplot as plt
import sqlite3
import sys

from predict_ca import model
from predict_ca import train_stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index_yh.html")

@app.route('/index3.html')
def index3():

    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select no2 from address"
    curs.execute(sql)
    rows = curs.fetchall() #no2 db 저장을 rows로 저장

    #백현동, BM별 분포 Pie Data
    curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 백현동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart2 = []
    print(chart_value)

    for i in chart_value:
        list_chart2.append(i[1])


    #판교동, BM별 분포 Pie Data
    curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 판교동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart3 = []
    print(chart_value)

    for i in chart_value:
        list_chart3.append(i[1])


    #삼평동, BM별 분포 Pie Data
    curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 삼평동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart4 = []
    print(chart_value)

    for i in chart_value:
        list_chart4.append(i[1])


    return render_template("index3.html", add2=rows, add22=rows, list_chart4=list_chart4, list_chart3=list_chart3, list_chart2=list_chart2)

@app.route('/index4.html')
def index4():

    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select no2 from address"
    curs.execute(sql)
    rows = curs.fetchall() #no2 db 저장을 rows로 저장


    #백현동, BM별 분포 Pie Data
    curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 백현동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart2 = []
    print(chart_value)

    for i in chart_value:
        list_chart2.append(i[1])


    #판교동, BM별 분포 Pie Data
    curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 판교동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart3 = []
    print(chart_value)

    for i in chart_value:
        list_chart3.append(i[1])


    #삼평동, BM별 분포 Pie Data
    curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 삼평동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart4 = []
    print(chart_value)

    for i in chart_value:
        list_chart4.append(i[1])

    return render_template("index4.html", add2=rows, add22=rows, list_chart4=list_chart4, list_chart3=list_chart3, list_chart2=list_chart2)


@app.route("/test", methods=["POST"])
def languages():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]
    value6 = request.form["itemid6"]

    print(value1)
    print(value4)
    print("데이터 넘어왔나?")

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute('SELECT * FROM sampledb where 1=1 and (?="" OR 세대수=?) and (?="선택하세요" OR 지역=?) and (?="선택하세요" OR 건물유형=?)',
                 (value6, value6, value4, value4, value5, value5))

    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    # print(rows)
    print("test2")

    curs.execute('SELECT * from piechart')
    chart_value = curs.fetchall()
    chart_value[0] = list(chart_value[0])
    print(chart_value[0])
    # print(type(chart_value[0][0]))

    # 모델링 샘플..
    x_test = [[int(value1), int(value2[0]), int(value3[0])]]
    print(x_test)
    # x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
    x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수'])
    model = tf.keras.models.load_model('model_v0.h5')
    model.summary()

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_x_test = norm(x_test)
    y_predict = model.predict(normed_x_test).tolist()
    print(y_predict[0])

    return jsonify(rows, chart_value[0], int(y_predict[0]))


@app.route("/test2", methods=["POST"])
def languages2():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]
    # value6 = request.form["itemid6"]
    #
    # print(value1)
    # print(value2)
    print("데이터 넘어왔나?")

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute('SELECT * FROM sampledb')

    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    # print(rows)
    print("test2")
    return jsonify(rows)


@app.route("/test3", methods=["POST"])
def languages3():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]
    # value6 = request.form["itemid6"]
    #
    print(value1)
    print(value2)
    print(value3)

    print("데이터 넘어왔나?")

    testvalue = value1

    if value2 !="":
        testvalue += " " + value2

    if value3 != "":
        testvalue += " " + value3

    if value4 != "":
        testvalue += " " + value4

    print(testvalue)

    if value5 == "시설 구분":
        testvalue_2 = "%"
        sql = "SELECT * FROM sampledb where 주소 like '%" + testvalue + "%'" #시설 구분(초기값)일때는 BM구분 없이 모든 db를 끌고옴
    else:
        testvalue_2 = value5
        sql = "SELECT * FROM sampledb where 주소 like '%" + testvalue + "%' and BM구분 LIKE '" + testvalue_2 + "'" #시설 구분에서 BM을 선택하면 그 BM에 대해서도 같이 db를 데려옴

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute(sql) #위의 BM구분에 대한 상황들을 처리해주기 위해 sql로 설정
    print(sql)
    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    print(rows)
    # print(rows)
    print("test3")
    return jsonify(rows)


@app.route("/pie1", methods=["POST"])
def piepie1():
    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute(
        'select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 삼평동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    chart_value = curs.fetchall()
    list_chart = []
    print(chart_value)

    for i in chart_value:
        list_chart.append(i[1])

    print(list_chart)

    return jsonify(list_chart)


@app.route('/add', methods=['POST'])
def add():
    value1 = request.form['x_py']
    print(value1)
    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select no3 from address where no2='" + value1 + "'"
    print(sql)
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)
    return jsonify(rows, value1)

@app.route("/predict", methods=["POST"])
def predict():
    x_test = [[request.form["itemid1"], request.form["itemid2"], request.form["itemid3"]]]
    # x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
    x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수'])

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    normed_x_test = norm(x_test)

    y_predict = model.predict(normed_x_test).tolist()
    print(y_predict[0])
    return jsonify(y_predict[0])

if __name__ == "__main__":
    app.run()