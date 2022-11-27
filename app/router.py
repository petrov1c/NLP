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
    model_name = request.headers['model']
    res = []
    if 'w2v_model' in globals():
        if model_name in w2v_model:
            if w2v_model[model_name].model_init:
                res.append("w2v ({}): It works!".format(model_name))

    if 'fasttext_cc_model' in globals():
        if model_name in fasttext_cc_model:
            if fasttext_cc_model[model_name].model_init:
                res.append("fasttext ({}): It works!".format(model_name))

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
    model_name = request.headers['model']
    return jsonify(fasttext_cc_model[model_name].model_info(request.get_json()))


@app.route("/fasttext_cc/config", methods=["POST"])
def fasttext_cc_config():
    model_name = request.headers['model']
    fasttext_cc_model[model_name].config(request.get_json())
    return jsonify({'result': True})


@app.route("/fasttext_cc/fit", methods=["POST"])
def fasttext_cc_fit():
    model_name = request.headers['model']
    fasttext_cc_model[model_name].fit(request.get_json())
    return jsonify({'result': True})


@app.route("/fasttext_cc/predict", methods=["POST"])
def fasttext_cc_predict():
    model_name = request.headers['model']
    data = request.get_json()
    count = 3
    threshhold = 0.5
    debug = False

    if 'Количество' in data:
        count = data['Количество']
    if 'Порог' in data:
        threshhold = data['Порог']
    if 'Отладка' in data:
        debug = data['Отладка']

    return jsonify(fasttext_cc_model[model_name].predict(data['СтрокаПоиска'], count, threshhold, debug))


# Word2Vec
@app.route("/w2v/model_info", methods=["POST"])
def w2v_model_info():
    model_name = request.headers['model']
    return jsonify(w2v_model[model_name].model_info())


@app.route("/w2v/config", methods=["POST"])
def w2v_config():
    model_name = request.headers['model']
    w2v_model[model_name].config(request.get_json())
    return jsonify({'result': True})


@app.route("/w2v/fit", methods=["POST"])
def w2v_fit():
    model_name = request.headers['model']
    return jsonify(w2v_model[model_name].fit(request.get_json()))


@app.route("/w2v/predict", methods=["POST"])
def w2v_predict():
    model_name = request.headers['model']
    return jsonify(w2v_model[model_name].predict(request.get_json()))


def get_models():
    models = []
    if os.path.isfile('./data/global_config.json'):
        with open('./data/global_config.json', "r", encoding='UTF-8') as read_file:
            models = json.load(read_file)
    return models


USED_MODELS = get_models()

if 'w2v' in USED_MODELS:
    w2v_model = dict()
    for name in USED_MODELS['w2v']:
        w2v_model[name] = G2V(name)

if 'fasttext_cc' in USED_MODELS:
    fasttext_cc_model = dict()
    for name in USED_MODELS['fasttext_cc']:
        fasttext_cc_model[name] = SearchModel(name)

if 'bert' in USED_MODELS:
    bert_model = bert.load_model()

if 'fasttext_gs' in USED_MODELS:
    fasttext_gs_model = KeyVectored()
