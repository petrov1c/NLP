from app import app
from app import bert

from flask import request
from flask import render_template

@app.route('/')
@app.route('/index')
def index():
    user = { 'nickname': 'Мой друг' } # выдуманный пользователь
    posts = [ # список выдуманных постов
        {
            'author': { 'nickname': 'John' },
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': { 'nickname': 'Susan' },
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template("index.html",
        title = 'Home',
        user = user,
        posts = posts)

@app.route('/test')
def test():
    return "It works!"

@app.route("/bert_embedding", methods=["POST"])
def embed_bert_cls():
    return bert.embed_bert_cls(request.get_json(), True)