from app import bert
from app.w2v import G2V
from app.fasttext_gs import KeyVectored
from app.fasttext_cc import SearchModel

from app import app
from app import renext

import os.path
import json

from flask import request, jsonify

@app.route('/', methods=["POST", "GET"])
@app.route('/index', methods=["POST", "GET"])
def index():
    return test()

@app.route('/test')
def test():
    res = []
    if 'w2v_model' in globals():
        if hasattr(w2v_model, 'model'):
            res.append("w2v: It works!")
        else:
            res.append("w2v: модель не инициализирована")

    if 'bert_model' in globals():
        if hasattr(bert_model, 'model'):
            res.append("bert: It works!")
        else:
            res.append("bert: модель не инициализирована")

        if not hasattr(bert_model, 'embs'):
            res.append("bert: модель не инициализирована")

    if 'fasttext_gs_model' in globals():
        if hasattr(fasttext_gs_model, 'model'):
            res.append("fasttext_gs: It works!")
        else:
            res.append("fasttext_gs: модель не инициализирована")

    if 'fasttext_cc_model' in globals():
        if hasattr(fasttext_cc_model, 'model'):
            res.append("fasttext_cc: It works!")
        else:
            res.append("fasttext_cc: модель не инициализирована")

    return ' '.join(res)

@app.route('/re/run', methods=["POST", "GET"])
def re_run():
    return jsonify(renext.run(request.get_json()))

@app.route('/re/pipeline')
def re_pipeline():
    data = request.get_json()
    return jsonify(renext.pipeline(data['Пайплайн'], data['Данные']))

# Tuni-bert-2
@app.route("/bert_embedding", methods=["POST"])
def embed_bert_cls():
    return bert_model.embed_bert_cls(request.get_json()['СтрокаПоиска'], True)

@app.route("/init_bert_embeddings", methods=["POST"])
def init_embeddings():
    return bert_model.init_embeddings(request.get_json()['Данные'])

@app.route("/predict_bert", methods=["POST"])
def predict_bert():
    return bert_model.predict(request.get_json())

# Fasttext gensim
@app.route("/fasttext/model_info", methods=["POST"])
def fasttext_model_info():
    return jsonify(fasttext_gs_model.model_info())

@app.route("/fasttext/config", methods=["POST"])
def fasttext_config():
    fasttext_gs_model.config(request.get_json())
    return jsonify({'result': True})

@app.route("/fasttext/update", methods=["POST"])
def fasttext_update():
    return jsonify(fasttext_gs_model.update(request.get_json()['Данные']))

@app.route("/fasttext/predict", methods=["POST"])
def fasttext_predict():
    return jsonify(fasttext_gs_model.predict(request.get_json()))

# Fasttext facebook
@app.route("/fasttext_cc/model_info", methods=["POST"])
def fasttext_cc_model_info():
    return jsonify(fasttext_cc_model.model_info())

@app.route("/fasttext_cc/config", methods=["POST"])
def fasttext_cc_config():
    fasttext_cc_model.config(request.get_json())
    return jsonify({'result': True})

@app.route("/fasttext_cc/fit", methods=["POST"])
def fasttext_cc_fit():
    fasttext_cc_model.fit(request.get_json()['Данные'])
    return jsonify({'result': True})

@app.route("/fasttext_cc/predict", methods=["POST"])
def fasttext_cc_predict():
    data = request.get_json()
    count = 3
    threshhold = 0.5
    debug = False

    if 'Количество' in data:
        count=data['Количество']
    if 'Порог' in data:
        threshhold=data['Порог']
    if 'Отладка' in data:
        debug=data['Отладка']

    return jsonify(fasttext_cc_model.predict(data['СтрокаПоиска'], count, threshhold, debug))

# Word2Vec
@app.route("/w2v/model_info", methods=["POST"])
def w2v_model_info():
    return jsonify(w2v_model.model_info())

@app.route("/w2v/config", methods=["POST"])
def w2v_config():
    w2v_model.config(request.get_json())
    return jsonify({'result': True})

@app.route("/w2v/fit", methods=["POST"])
def w2v_fit():
    return jsonify(w2v.fit(w2v_model))

@app.route("/w2v/update", methods=["POST"])
def w2v_update():
    w2v_model.update(request.get_json())
    return jsonify({'result': True})

@app.route("/w2v/predict", methods=["POST"])
def w2v_predict():
    return jsonify(w2v_model.predict(request.get_json()))

def get_models():
    models = []
    if os.path.isfile('./data/global_config.json'):
        with open('./data/global_config.json', "r", encoding='UTF-8') as read_file:
            models = json.load(read_file)
    return models

USED_MODELS = get_models()

if 'bert' in USED_MODELS:
    bert_model = bert.load_model()

if 'w2v' in USED_MODELS:
    w2v_model = G2V()

if 'fasttext_gs' in USED_MODELS:
    fasttext_gs_model = KeyVectored()

if 'fasttext_cc' in USED_MODELS:
    fasttext_cc_model = SearchModel()