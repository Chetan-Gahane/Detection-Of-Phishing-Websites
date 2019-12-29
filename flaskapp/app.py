
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import newtrain
 
app = Flask(__name__)
 
@app.route('/')
def home(x):
    return x
    
 
@app.route('/login', methods=['POST'])
def do_admin_login():
    url_new=request.form['username']
    x=newtrain.main(url_new)
    return home(x)
 
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)