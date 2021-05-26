from flask import Flask, request, session,g,redirect,url_for, abort, flash, render_template, jsonify, json
import pandas as pd
import sqlite3

app = Flask(__name__)

@app.route('/')
def yeon():
    return render_template("login_test.html")
@app.route('/index.html')
def yeon1():
    return render_template("index.html")
@app.route('/password.html')
def yeon2():
    return render_template("password.html")
@app.route('/test.html')
def yeon3():
    return render_template("test.html")

@app.route('/test2.html')
def yeon4():
    return render_template("test2.html")

@app.route('/login.html')
def yeon5():
    return render_template("login.html")

@app.route('/test', methods=['POST'])
def languages():
    value1 = request.form['itemid']
    print(value1)

    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()
    curs.execute("SELECT * FROM test1 WHERE 용도='상가'") #test1에 있는 값을 조건 없이 불러옴

    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    print(rows)
    print("test2")
    #
    #
    # print(rows)
    #
    return jsonify(rows)

# @app.route('/test2', methods=['POST'])
# def languages2():
#     value2 = request.form['itemid']
#     print(value2)
#
#     conn = sqlite3.connect('trest.db')
#     curs = conn.cursor()
#     #
#     curs.execute("select * from address") #test1에 있는 값을 조건 없이 불러옴
#     # row_headers = [x[0] for x in curs.description]
#     rows = curs.fetchall()
#     #
#     #
#     # print(rows)
#     #
#     return jsonify(rows)

if __name__=='__main__':
    app.run()