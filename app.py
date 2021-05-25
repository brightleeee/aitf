import pandas as pd
from flask import Flask, render_template, request
import main


app = Flask(__name__)


@app.route('/')
def index():

    return render_template('index.html')

@app.route('/form')
def student():
    # ROOT URL 로 접근했을 때 8_sending_form_data.html 을 렌더링한다.
    #  return render_template('sending_form_data.html')

    return render_template('sending_form_data.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
    # result URL 로 접근했을 때, form 데이터는 request 객체를 이용해 전달받고
    # 이를 다시 8_sending_form_data_result.html 렌더링하면서 넘겨준다.
    if request.method == 'POST':
        result = request.form
        result2 = dict(request.form)
        result3 = pd.DataFrame(result, index=[0])
        result4 = result3.to_html()

        print(result3)
        print(result3.values)
        print(result3)


        return render_template("sending_form_data_result2.html", tables = result3, titles = '2')

    else:
        return render_template('new.html')

@app.route('/ajax')
def main1():
    results = {}
    return render_template('index.html', results = sheet3)

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/new')
def new():
    return render_template('new.html')

@app.route('/old')
def old():
    return render_template('old.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
