from flask import Flask,render_template, request,jsonify,send_file
from detectingWords import Data
from classifier import load_classfication_pred_res
from trainerclassifier import texts,add_data_and_retrain_model
from naivebase import get_accuracy
from linear_classifier import get_accuracy2
from pythonapi import callapi
import json
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("abc.html")


@app.route('/process',methods= ['POST'])
def process():
    #receive and processed that data for prediction
    data =str(request.form['data1']) 
    abc = Data(data)
    abc.preprocesstext=abc.convert_lower_to_upper()
    
    abc.removing_commonword()
    result=[]
    position=[]
    profane=0
    for i in range(len(abc.preprocesstext)):
        result.append(load_classfication_pred_res(abc.preprocesstext[i]))
    
    for i in range(len(result)):
        if result[i]==0:
            profane=1
            position.append(abc.orignalText.find(abc.preprocesstext[i]))
    if profane == 0:
        return "There is No Obsence word in text"
    else:
        stri = ""
        for i in range(len(position)):
            stri = stri + str(position[i])+" "
        return "Obsense words are start at index "+stri+"in the string"
    return data
@app.route('/callapi',methods= ['POST'])
def process1():
    #receive and processed that data for prediction
    data =str(request.form['data1'])
    retur=callapi(data)
    json_string = retur
    parsed_json = json.loads(json_string)
    output="Badwords: "
    badwords =parsed_json["bad-words-list"]
    for a in badwords:
        output= output+a+" "
    isbad=parsed_json["is-bad"]
    output= output + "+IsBad: "+str(isbad)
    return output
@app.route('/nb',methods= ['POST'])
def naivebaseaccuracy():
    return get_accuracy()

@app.route('/lc',methods= ['POST'])
def Linear_Regressionaccuracy():
    return get_accuracy2()

@app.route('/addData',methods= ['POST'])
def process3():
    #receive and processed that data for prediction
    data =str(request.form['data1'])
    data=add_data_and_retrain_model(data.split())
    return data
if __name__ == "__main__":
    app.run(debug=True)