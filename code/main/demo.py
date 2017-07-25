from flask import Flask
app = Flask(__name__)
app.secret_key='LongLiveTheCaesar1239988KingOfKings'
from flask import request
from flask import render_template
from flask import _app_ctx_stack
from flask import g
from flask import session

@app.route("/test", methods=['GET', 'POST'] )
def test():
    print request.args
    inp = request.args.get('text')
    portmanteau_inputs = inp.split(',')
    #shakepeare_inputs = inp.strip()
    print portmanteau_inputs
    # TO DO: insert logic for portmanteau prediction / shakeeare style prediction
    return inp

@app.route("/portmanteau",methods=['GET'])
def hello():
    return render_template("hello.html")

@app.route("/portmanteau",methods=['POST'])
def hello_post():
    text=request.form['text']
    text2=request.form['text2']
    text=text.encode('utf8')
    text2=text2.encode('utf8')
    text=text.lower()
    text2=text2.lower()
    if len(text)==0 or not text.isalpha():
        text="alpha"
    elif len(text)>30:
        text=text[:30]
    if len(text2)==0 or not text2.isalpha():
        text2="beta"
    elif len(text2)>30:
        text2=text2[:30]
    inBuf=open("demoLogs/inBuf.txt","a")
    inBuf.write(text+","+text2+"\n")
    inBuf.close()
    answers=bed.query(text,text2,predictor)
    answers=[str(i+1)+". "+answer for i,answer in enumerate(answers)]
    headerString="<h1>The top 5 suggestions for a portmanteau are as below</h1>"
    feedbackString=open("templates/feedbackForm.html").read()
    answerString=headerString+"<br/>".join(answers)+"<br/>"+feedbackString
    session['text']=text
    session['text2']=text2
    return answerString

@app.route("/portmanteau_feedback",methods=['POST'])
def feedback():
    if 'text' in session:
        text=session['text']
        text2=session['text2']
        session.pop('text',None)
        session.pop('text2',None)
    
    feedback=request.form['feedback']
    outBuf=open("demoLogs/outBuf.txt","a")
    outBuf.write(text+","+text2+","+feedback+"\n")
    outBuf.close()
    return "Feedback Received"
