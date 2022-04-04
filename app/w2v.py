import gensim                          # Гугл 2013 год
from razdel import tokenize            # проект Natasha
from multiprocessing import cpu_count  # для распараллеливания

import json
import os.path

from flask import jsonify

class G2V:
    def __init__(self):
        if os.path.isfile('./data/w2v.txt'):
            self.model = gensim.models.KeyedVectors.load_word2vec_format('./data/w2v.txt', binary=True)
        elif os.path.isfile('./data/goods.json'):
            with open('./data/goods.json', "r", encoding='UTF-8') as read_file:
                sentences = json.load(read_file)

            window_size = 3
            sg = 0
            epoch = 100
            corpus = 'goods'

            if corpus == 'goods':
                data = [sentence.split() for sentence in sentences['Данные']]
            else:
                data = [[i.text for i in tokenize(sentence.lower())] for sentence in sentences['Данные']]

            model = gensim.models.Word2Vec(data, window=window_size, sg=sg, min_count=3, epochs=epoch, workers=cpu_count())
            model.wv.save_word2vec_format('./data/w2v.txt', binary=True)

            self.model = model.wv

    def update(self, data):
        if os.path.isfile('./data/w2v.txt'):
            os.remove('./data/w2v.txt')
        with open('./data/goods.json', "w", encoding='utf-8') as write_file:
            json.dump(data, write_file)

    def predict(self, data):
        if hasattr(self, 'model'):
            if 'Количество' in data:
                count = data['Количество']
            else:
                count = 3

            positive = [i for i in data['Данные'] if i in self.model.key_to_index]
            if len(positive):
                rez_data = self.model.most_similar(positive=positive, topn=count)
                rez_data = {i[0]:i[1] for i in rez_data}
                return jsonify({'result': True, 'data': rez_data})
            else:
                return jsonify({'result': False, 'error': 'товаров нет в словаре'})
        else:
            return jsonify({'result': False, 'error': 'Модель не инициализирована'})

def load_model():
    return G2V()