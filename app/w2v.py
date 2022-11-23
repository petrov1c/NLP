import gensim  # Гугл 2013 год
import threading  # Для запуска фонового обучения
from multiprocessing import cpu_count  # для распараллеливания

import os.path
import json
import time
from datetime import datetime


class G2V:
    def __init__(self, name):
        start_time = time.time()
        self.name = name
        self.data_file = './data/w2v/{}.txt'.format(name)
        self.config_file = './data/w2v/{}_config.json'.format(name)
        self.model_init = False
        self.training = False
        self.fit_time = 0
        self.date_training = datetime.isoformat(datetime(1, 1, 1))

        if os.path.isfile(self.data_file):
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.data_file, binary=True)
            self.model_init = True

        self.date_init = datetime.isoformat(datetime.now())
        self.startup_time = round(time.time() - start_time, 3)

    def model_info(self):

        info = dict()
        info['модель создана'] = self.model_init
        if self.model_init:
            info['дата создания'] = self.date_init
            info['время запуска'] = self.startup_time
            info['размер словаря'] = len(self.model)

        info['идет обучение'] = self.training
        info['дата обучения'] = self.date_training
        info['время обучения'] = self.fit_time

        if os.path.isfile(self.config_file):
            with open(self.config_file, "r", encoding='UTF-8') as read_file:
                info['конфигурационный файл'] = json.load(read_file)
        else:
            info['конфигурационный файл'] = 'отсутствует'

        return info

    def fit(self, x):
        if hasattr(self, 'training') and self.training:
            return {'result': False, 'error': 'Обучение уже запущено'}
        else:
            thread_fit = threading.Thread(target=self.fit_model, kwargs={'x': x})
            thread_fit.start()
            return {'result': True}

    def fit_model(self, x):

        self.training = True

        start_time = time.time()

        window_size = 3
        emb_size = 100
        sg = 0
        epoch = 100

        if os.path.isfile(self.config_file):
            with open(self.config_file, "r", encoding='UTF-8') as read_file:
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

        data = [sentence.split() for sentence in x]

        model = gensim.models.Word2Vec(data, window=window_size, sg=sg, min_count=3, vector_size=emb_size, epochs=epoch,
                                       workers=cpu_count())
        model.wv.save_word2vec_format(self.data_file, binary=True)

        self.__init__(self.name)
        self.fit_time = round(time.time() - start_time, 3)
        self.date_training = datetime.isoformat(datetime.now())

    def config(self, data):
        with open(self.config_file, "w", encoding='UTF-8') as write_file:
            json.dump(data, write_file)

        # ToDo Переделать в асинхронный режим
        self.__init__(self.name)

    def predict(self, data):
        if self.model_init:
            if 'Количество' in data:
                count = data['Количество']
            else:
                count = 3

            positive = [i for i in data['Данные'] if i in self.model.key_to_index]
            if len(positive):
                rez_data = self.model.most_similar(positive=positive, topn=count)
                rez_data = [{i[0]: i[1]} for i in rez_data]
                return {'result': True, 'data': rez_data}
            else:
                return {'result': False, 'error': 'товаров нет в словаре'}
        else:
            return {'result': False, 'error': 'Модель не инициализирована'}
