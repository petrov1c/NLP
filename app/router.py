from app import bert
from app import w2v
from app import fasttext
from app import app
from app import renext

from flask import request, jsonify

bert_model = bert.load_model()
w2v_model = w2v.load_model()
fasttext_model = fasttext.load_model()

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

@app.route('/re/run', methods=["POST", "GET"])
def re_run():
    return jsonify(renext.run(request.get_json()))

@app.route('/re/pipeline')
def re_pipeline():
    data = request.get_json()
    return jsonify(renext.pipeline(data['Пайплайн'], data['Данные']))

@app.route("/bert_embedding", methods=["POST"])
def embed_bert_cls():
    return bert_model.embed_bert_cls(request.get_json()['СтрокаПоиска'], True)

@app.route("/init_bert_embeddings", methods=["POST"])
def init_embeddings():
    return bert_model.init_embeddings(request.get_json()['Данные'])

@app.route("/predict_bert", methods=["POST"])
def predict_bert():
    return bert_model.predict(request.get_json())

@app.route("/fasttext/model_info", methods=["POST"])
def fasttext_model_info():
    return jsonify(fasttext_model.model_info())

@app.route("/fasttext/config", methods=["POST"])
def fasttext_config():
    fasttext_model.config(request.get_json())
    return jsonify({'result': True})

@app.route("/fasttext/update", methods=["POST"])
def fasttext_update():
    return jsonify(fasttext_model.update(request.get_json()['Данные']))

@app.route("/fasttext/predict", methods=["POST"])
def fasttext_predict():
    return jsonify(fasttext_model.predict(request.get_json()))

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