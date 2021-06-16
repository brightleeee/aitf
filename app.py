from typing import Any


from flask import (
    Flask,
    request,
    session,
    g,
    redirect,
    url_for,
    abort,
    flash,
    render_template,
    jsonify,
    json,
)
import pandas as pd
import sqlite3

app = Flask(__name__)


@app.route("/")
def yeon():
    return render_template("login_test.html")


@app.route("/index.html")
def yeon1():
    return render_template("index.html")


@app.route("/password.html")
def yeon2():
    return render_template("password.html")


@app.route("/test.html")
def yeon3():
    return render_template("test.html")


@app.route("/test3.html")
def yeon4():
    return render_template("test3.html")


@app.route("/login.html")
def yeon5():
    return render_template("login.html")


@app.route("/test", methods=["POST"])
def languages():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]

    print(value1)
    print(value2)
    print("데이터 넘어왔나?")

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute('SELECT * FROM test1 where 1=1 and (?="" OR 면적=?) and (?="주거용 / 상업용" OR 용도=?)',
                 (value1,value1,value4,value4))  # test1에 있는 값을 조건 없이 불러옴 or조건으로 입력값이 있는 것만 조회함

    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    print(rows)
    print("test2")
    return jsonify(rows)


if __name__ == "__main__":
    app.run()
