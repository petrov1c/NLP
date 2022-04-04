from app import bert
from app import w2v
from app import app

from flask import request
from flask import render_template

bert_model = bert.load_model()
w2v_model = w2v.load_model()

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
    res = []
    if hasattr(w2v_model, 'model'):
        res.append("w2v: It works!")
    else:
        res.append("w2v: модель не инициализирована")

    if hasattr(bert_model, 'model'):
        res.append("bert: It works!")
    else:
        res.append("bert: модель не инициализирована")

    return ' '.join(res)

@app.route('/re')
def re():
    return bert.regex(request.get_json())

@app.route("/bert_embedding", methods=["POST"])
def embed_bert_cls():
    return bert_model.embed_bert_cls(request.get_json()['СтрокаПоиска'], True)

@app.route("/init_bert_embeddings", methods=["POST"])
def init_embeddings():
    return bert_model.init_embeddings(request.get_json()['Данные'])

@app.route("/update_w2v", methods=["POST"])
def update_w2v():
    return w2v_model.update(request.get_json())

@app.route("/predict_bert", methods=["POST"])
def predict_bert():
    return bert_model.predict(request.get_json())

@app.route("/predict_w2v", methods=["POST"])
def predict_w2v():
    return w2v_model.predict(request.get_json())