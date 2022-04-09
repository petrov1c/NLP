import gensim                          # Гугл 2013 год
from razdel import tokenize            # проект Natasha
import threading                       # Для запуска фонового обучения
from multiprocessing import cpu_count  # для распараллеливания

import json
import os.path
from datetime import datetime
import time

class G2V:
    def __init__(self):
        if os.path.isfile('./data/w2v.txt'):
            self.model = gensim.models.KeyedVectors.load_word2vec_format('./data/w2v.txt', binary=True)
            self.model_init = True
            self.date_init = datetime.isoformat(datetime.now())

        elif os.path.isfile('./data/w2v_data.json'):
            self.fit()

    def model_info(self):

        info = {}
        if hasattr(self, 'model'):
            info['model'] = 'модель создана'
        else:
            info['model'] = 'модель не создана'

        if hasattr(self, 'model_init'):
            info['model_init'] = self.model_init

        if hasattr(self, 'date_init'):
            info['date_init'] = self.date_init

        if hasattr(self, 'training_time'):
            info['training_time'] = self.training_time

        if hasattr(self, 'training'):
            info['training'] = self.training

        if os.path.isfile('./data/w2v_config.json'):
            with open('./data/w2v_config.json', "r", encoding='UTF-8') as read_file:
                info['config'] = json.load(read_file)
        else:
            info['config'] = 'отсутствует'

        return info

    def fit(self):

        self.model_init = False
        self.training = True

        if os.path.isfile('./data/w2v_data.json'):
            with open('./data/w2v_data.json', "r", encoding='UTF-8') as read_file:
                sentences = json.load(read_file)

            window_size = 3
            emb_size = 100
            sg = 0
            epoch = 100
            corpus = 'goods'

            if os.path.isfile('./data/w2v_config.json'):
                with open('./data/w2v_config.json', "r", encoding='UTF-8') as read_file:
                    w2v_config = json.load(read_file)
                    if isinstance(w2v_config, dict):
                        if 'window_size' in w2v_config:
                            window_size = w2v_config['window_size']
                        if 'emb_size' in w2v_config:
                            emb_size = w2v_config['emb_size']
                        if 'sg' in w2v_config:
                            sg = w2v_config['sg']
                        if 'epoch' in w2v_config:
                            epoch = w2v_config['epoch']
                        if 'corpus' not in w2v_config:
                            corpus = w2v_config['corpus']

            if corpus == 'goods':
                data = [sentence.split() for sentence in sentences['Данные']]
            else:
                #ToDo тексты очищать через re перед передачей их в токенайзер
                data = [[i.text for i in tokenize(sentence.lower())] for sentence in sentences['Данные']]

            start_time = time.time()

            model = gensim.models.Word2Vec(data, window=window_size, sg=sg, min_count=3, size = emb_size, epochs=epoch, workers=cpu_count())
            model.wv.save_word2vec_format('./data/w2v.txt', binary=True)

            self.model = model.wv
            self.model_init = True
            self.date_init = datetime.isoformat(datetime.now())
            self.training_time = time.time() - start_time
            self.training = False

    def config(self, data):
        with open('./data/w2v_config.json', "w", encoding='UTF-8') as write_file:
            json.dump(data, write_file)

    def update(self, data):
        if os.path.isfile('./data/w2v.txt'):
            os.remove('./data/w2v.txt')

        with open('./data/w2v_data.json', "w", encoding='UTF-8') as write_file:
            json.dump(data, write_file)

    def predict(self, data):
        if self.model_init:
            if 'Количество' in data:
                count = data['Количество']
            else:
                count = 3

            positive = [i for i in data['Данные'] if i in self.model.key_to_index]
            if len(positive):
                rez_data = self.model.most_similar(positive=positive, topn=count)
                rez_data = {i[0]:i[1] for i in rez_data}
                return {'result': True, 'data': rez_data}
            else:
                return {'result': False, 'error': 'товаров нет в словаре'}
        else:
            return {'result': False, 'error': 'Модель не инициализирована'}

def fit(model):

    if hasattr(model, 'training') and model.training:
        return {'result': False, 'error': 'Обучение уже запущено'}
    else:
        thread_fit = threading.Thread(target=model.fit)
        thread_fit.start()
        return {'result': True}

def load_model():
    return G2V()