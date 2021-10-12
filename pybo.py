# from typing import Any
from flask import Flask, request, session,g,redirect,url_for, abort, flash, render_template, jsonify, json
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import sys
import numpy as np
from predict_ca_master2 import model, model2, model3, model4
from predict_ca_master2 import train_stats, train_stats2, train_stats3, train_stats4
from predict_ca_master3 import model5, model6, model7, model8
from predict_ca_master3 import train_stats5, train_stats6, train_stats7, train_stats8

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import math

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index_yh.html")


@app.route('/index3.html')
def index3():

    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select distinct 지사 from groups" #distinct 중복값 제거
    curs.execute(sql)
    rows = curs.fetchall()

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


    return render_template("index3.html", g22=rows, list_chart4=list_chart4, list_chart3=list_chart3, list_chart2=list_chart2)


@app.route('/index4.html')
def index4():

    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select distinct no2 from address"
    curs.execute(sql)
    rows = curs.fetchall() #no2 db 저장을 rows로 저장

    sql2 = "select distinct 지사 from groups" #distinct 중복값 제거
    curs.execute(sql2)
    rows2 = curs.fetchall()

    #오포읍, BM별 분포 Pie Data
    # curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 백현동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    curs.execute('select sum(무선),sum(전송),sum(전용),sum(초고속) from sampledb where 주소 like "%광주시 오포읍 신현리%" ')
    chart_value = curs.fetchall()
    list_chart1 = []
    print(chart_value)

    for i in chart_value[0]:
        list_chart1.append(i)
    print(list_chart1)

    #백현동, BM별 분포 Pie Data
    # curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 백현동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    curs.execute('select sum(무선),sum(전송),sum(전용),sum(초고속) from sampledb where 주소 like "%성남시 분당구 백현동%" ')
    chart_value = curs.fetchall()
    list_chart2 = []
    print(chart_value)

    for i in chart_value[0]:
        list_chart2.append(i)


    #판교동, BM별 분포 Pie Data
    # curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 판교동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    curs.execute('select sum(무선),sum(전송),sum(전용),sum(초고속) from sampledb where 주소 like "%성남시 분당구 판교동%" ')
    chart_value = curs.fetchall()
    list_chart3 = []
    print(chart_value)

    for i in chart_value[0]:
        list_chart3.append(i)


    #삼평동, BM별 분포 Pie Data
    # curs.execute('select BM구분, count(*) from sampledb where substr(주소, 5, 11) = "성남시 분당구 삼평동" and BM구분 IS NOT NULL group by BM구분 ORDER BY BM구분')
    curs.execute('select sum(무선),sum(전송),sum(전용),sum(초고속) from sampledb where 주소 like "%성남시 분당구 삼평동%" ')
    chart_value = curs.fetchall()
    list_chart4 = []
    print(chart_value)

    for i in chart_value[0]:
        list_chart4.append(i)

    return render_template("index4.html", add2=rows, g22=rows2, list_chart4=list_chart4, list_chart3=list_chart3, list_chart2=list_chart2, list_chart1=list_chart1)


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

    #curs.execute('SELECT * from piechart')
    #chart_value = curs.fetchall()
    #chart_value[0] = list(chart_value[0])
    #print(chart_value[0])

    #return jsonify(rows, chart_value[0])

    if value5 == "빌딩":
        x_test = [[int(value1), int(value2), int(value3)]]
        print(x_test)
        # x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
        x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수'])
        model = tf.keras.models.load_model('model_0923_v1_cho.h5')
        model2 = tf.keras.models.load_model('model_0923_v1_jeon.h5')
        model3 = tf.keras.models.load_model('model_0923_v1_moo.h5')
        model4 = tf.keras.models.load_model('model_0923_v1_gwang.h5')

        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        def norm2(x):
            return (x - train_stats2['mean']) / train_stats2['std']

        def norm3(x):
            return (x - train_stats3['mean']) / train_stats3['std']

        def norm4(x):
            return (x - train_stats4['mean']) / train_stats4['std']

        normed_x_test = norm(x_test)

        y_predict = np.round(model.predict(normed_x_test), 0).tolist()

        normed_x_test2 = norm2(x_test)
        y_predict1 = np.round(model2.predict(normed_x_test2), 0).tolist()

        normed_x_test3 = norm3(x_test)
        y_predict2 = np.round(model3.predict(normed_x_test3), 0).tolist()

        normed_x_test4 = norm4(x_test)
        y_predict3 = np.round(model4.predict(normed_x_test4), 0).tolist()

        print(y_predict)
        print(y_predict1)
        print(y_predict2)
        print(y_predict3)
        print(y_predict[0])
        y_predict.append(y_predict1[0]) #전용 [22]
        y_predict.append(y_predict3[0]) #광전화
        y_predict.append(y_predict2[0]) #무선
        resulttest = 0
        for i in y_predict:
            print(i[0])
            resulttest += i[0]
        print("y_predict:"+str(y_predict))
        
    elif value5 == "상가":
        x_test = [[int(value1), int(value2), int(value3)]]
        print(x_test)
        # x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
        x_test = pd.DataFrame(x_test, columns=['연면적', '최대층수', '최저층수'])
        model5 = tf.keras.models.load_model('model_0923_v2_cho.h5')
        model6 = tf.keras.models.load_model('model_0923_v2_jeon.h5')
        model7 = tf.keras.models.load_model('model_0923_v2_moo.h5')
        model8 = tf.keras.models.load_model('model_0923_v2_gwang.h5')
    
        def norm5(x):
            return (x - train_stats5['mean']) / train_stats5['std']
    
        def norm6(x):
            return (x - train_stats6['mean']) / train_stats6['std']
    
        def norm7(x):
            return (x - train_stats7['mean']) / train_stats7['std']
    
        def norm8(x):
            return (x - train_stats8['mean']) / train_stats8['std']
    
        normed_x_test5 = norm5(x_test)
    
        y_predict5 = np.round(model5.predict(normed_x_test5), 0).tolist()
    
        normed_x_test6 = norm6(x_test)
        y_predict6 = np.round(model6.predict(normed_x_test6), 0).tolist()
    
        normed_x_test7 = norm7(x_test)
        y_predict7 = np.round(model7.predict(normed_x_test7), 0).tolist()
    
        normed_x_test8 = norm8(x_test)
        y_predict8 = np.round(model8.predict(normed_x_test8), 0).tolist()
    
        print(y_predict5)
        print(y_predict6)
        print(y_predict7)
        print(y_predict8)
        print(y_predict5[0])
        y_predict5.append(y_predict6[0])  # 전용 [22]
        y_predict5.append(y_predict7[0])  # 광전화
        y_predict5.append(y_predict8[0])  # 무선
        resulttest = 0
        for i in y_predict5:
            print(i[0])
            resulttest += i[0]
        print("y_predict:" + str(y_predict5))

        y_predict = y_predict5

    return jsonify(rows, y_predict, resulttest)

@app.route("/test2", methods=["POST"])
def languages2():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]
    # value6 = request.form["itemid6"]
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
    # value9 = request.form["itemid9"]


    print(value2)
    print(value3)
    # print(value9)
    print("데이터 넘어왔나?")

    testvalue = value1
    sql2 = "SELECT substr(주소, 5, 11), sum(무선),sum(전송),sum(전용),sum(초고속) FROM sampledb group by substr(주소, 5, 11)"
    if value2 not in ("","선택하세요"):
        testvalue += " " + value2
        sql2 = "SELECT substr(주소, 5, 11), sum(무선),sum(전송),sum(전용),sum(초고속) FROM sampledb where (주소 like '%" + testvalue + "%' group by substr(주소, 5, 11)"

    if  value3 not in ("","선택하세요"):
        testvalue += " " + value3
        sql2 = "SELECT substr(주소, 5, 11), sum(무선),sum(전송),sum(전용),sum(초고속) FROM sampledb where 주소 like '%" + testvalue + "%' group by substr(주소, 5, 11)"

    if value4 != "":
        testvalue += " " + value4
    else:
        testvalue += "%"

    print(testvalue)

    if value5 == "시설 구분":
        testvalue_2 = "%"
        sql = "SELECT * FROM sampledb where 주소 like '%" + testvalue + "' or 주소 like '%" + testvalue + "번지'"  #시설 구분(초기값)일때는 BM구분 없이 모든 db를 끌고옴
        sql2 = "SELECT substr(주소, 5, 11), sum(무선),sum(전송),sum(전용),sum(초고속) FROM sampledb where 주소 like '%" + testvalue + "' or 주소 like '%" + testvalue + "번지' group by substr(주소, 5, 11)"
    else:
        testvalue_2 = value5
        sql = "SELECT * FROM sampledb where (주소 like '%" + testvalue + "' or 주소 like '%" + testvalue + "번지') and BM구분 LIKE '" + testvalue_2 + "'" #시설 구분에서 BM을 선택하면 그 BM에 대해서도 같이 db를 데려옴
        sql2 = "SELECT substr(주소, 5, 11), sum(무선),sum(전송),sum(전용),sum(초고속) FROM sampledb where (주소 like '%" + testvalue + "' or 주소 like '%" + testvalue + "번지') and BM구분 LIKE '" + testvalue_2 + "' group by substr(주소, 5, 11)"


    print("sql2 : " + sql2)

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute(sql) #위의 BM구분에 대한 상황들을 처리해주기 위해 sql로 설정
    print(sql)
    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    print(rows)

    curs.execute(sql2)  # 위의 BM구분에 대한 상황들을 처리해주기 위해 sql로 설정
    print(sql2)
    # row_headers = [x[0] for x in curs.description]
    rows2 = curs.fetchall()
    print(rows2)

    # print(rows)
    print("test3")


    return jsonify(rows, rows2)


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

@app.route('/group2', methods=['POST'])
def group_change():
    value1 = request.form['x_py']
    print(value1)
    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select distinct 지점 from groups where 지사='" + value1 + "'"
    print(sql)
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)
    return jsonify(rows, value1)

@app.route('/group3', methods=['POST'])
def group_change2():
    value1 = request.form['x_py']
    print(value1)
    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select 국사 from groups where 지점='" + value1 + "'"
    print(sql)
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)
    return jsonify(rows, value1)

@app.route("/predict", methods=["POST"])
def predict():
    x_test = [[request.form["itemid1"], request.form["itemid2"], request.form["itemid3"]]]
    # x_test = pd.DataFrame(x_test, columns=['연면적', '세대수'])
    x_test = pd.DataFrame(x_test, columns=['연면적', '최고층수', '최저층수'])

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    normed_x_test = norm(x_test)

    y_predict = model.predict(normed_x_test).tolist()
    print(y_predict[0])
    return jsonify(y_predict[0])


if __name__ == "__main__":
    app.run()