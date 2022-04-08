from app import bert
from app import w2v
from app import app

from flask import request, jsonify

bert_model = bert.load_model()
w2v_model = w2v.load_model()

@app.route('/')
@app.route('/index')
def index():
    return test()

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

    if not hasattr(bert_model, 'embs'):
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

@app.route("/predict_bert", methods=["POST"])
def predict_bert():
    return bert_model.predict(request.get_json())

@app.route("/w2v/model_info", methods=["POST"])
def w2v_model_info():
    return jsonify(w2v_model.model_info())

@app.route("/w2v/config", methods=["POST"])
def w2v_config():
    w2v_model.config(request.get_json())
    return jsonify({'result': True})

@app.route("/w2v/fit", methods=["POST"])
def w2v_fit():
    w2v.fit(w2v_model)
    # w2v_model.fit()
    return jsonify({'result': True})

@app.route("/w2v/update", methods=["POST"])
def w2v_update():
    w2v_model.update(request.get_json())
    return jsonify({'result': True})

@app.route("/w2v/predict", methods=["POST"])
def w2v_predict():
    return jsonify(w2v_model.predict(request.get_json()))