import fasttext
import torch

import threading                       # Для запуска фонового обучения

import os.path
import re
import json
import time
from datetime import datetime


class SearchModel:
    def __init__(self):
        start_time = time.time()
        self.model_init = False
        self.training = False
        self.fit_time = 0
        self.date_training = ''

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.re = re.compile(r'[^а-яА-ЯёЁ0-9a-zA-Z @!?,.|/:;\'""*&@#$№%\[\]{}()+\-$]')

        if os.path.isfile('./data/fasttext_cc_config.json'):
            with open('./data/fasttext_cc_config.json', "r", encoding='UTF-8') as read_file:
                config = json.load(read_file)

            if 'model' in config:
                if os.path.isfile(config['model']):
                    self.model = fasttext.load_model(config['model'])
                    self.model_init = True
                    self.date_init = datetime.isoformat(datetime.now())

                    self.embs_is_load = False

                if os.path.isfile('./data/fasttext_cc_embs.json'):
                    with open('./data/fasttext_cc_embs.json', "r") as read_file:
                        loaded_data = json.load(read_file)

                    self.data = loaded_data['data']
                    self.embs = torch.zeros((len(loaded_data['embs']), self.model.get_dimension())).to(self.device)
                    for idx, key in enumerate(loaded_data['embs']):
                        self.embs[idx] = torch.Tensor(loaded_data['embs'][key]).to(self.device)

                    self.embs_is_load = True
                    self.date_load_embs = datetime.isoformat(datetime.now())
                    self.time_load_embs = round(time.time() - start_time, 3)

            if 'pattern' in config:
                self.re = re.compile(r'{}'.format(config['pattern']))

        self.startup_time = round(time.time() - start_time, 3)

    def preprocess(self, text):
        text = text.lower().strip()
        return self.re.sub('', text)

    def fit(self, x):
        if self.training:
            return {'result': False, 'error': 'Обучение уже запущено'}
        else:
            thread_fit = threading.Thread(target=self.fit_model, kwargs={'x': x})
            thread_fit.start()
            return {'result': True}

    def fit_model(self, x):
        self.training = True
        self.embs_is_load = False

        start_time = time.time()

        self.data = []
        for item in x:
            line = self.preprocess(item['Наименование'])
            if line != '':
                self.data.append({'code': item['Код'], 'text': line})

        self.embs = torch.zeros((len(self.data), self.model.get_dimension())).to(self.device)
        for idx, value in enumerate(self.data):
            self.embs[idx] = torch.Tensor(self.model.get_sentence_vector(value['text'])).to(self.device)

        self.training = False
        self.embs_is_load = True
        self.date_load_embs = datetime.isoformat(datetime.now())
        self.time_load_embs = round(time.time() - start_time, 3)

        embs = {data['code']: self.embs[idx].tolist() for idx, data in enumerate(self.data)}
        with open('./data/fasttext_cc_embs.json', "w") as write_file:
            save_data = {'data': self.data, 'embs': embs}
            json.dump(save_data, write_file)

    def predict(self, text, top=3, threshold=0.5, debug=False):
        if not self.model_init:
            return {'result': False, 'error': 'Создайте модель'}
        elif not self.embs_is_load:
            return {'result': False, 'error': 'Загрузите эмбеддинги'}

        text = self.preprocess(text)
        if text == '':
            return {'data': [], 'text': text}
        else:
            search_emb = torch.Tensor(self.model.get_sentence_vector(text)).to(self.device)
            search_emb = search_emb.tile((self.embs.shape[0], 1)).to(self.device)

            rez = torch.cosine_similarity(search_emb, self.embs)
            sort_idx = rez.argsort(descending=True)[:top].tolist()
            if threshold > 0:
                thresh_idx = rez > threshold
                data = [{self.data[idx]['code']: rez[idx].item()} for idx in sort_idx if thresh_idx[idx]]
                if debug:
                    debug_data = [[self.data[idx]['code'], self.data[idx]['text'], rez[idx].item(), self.embs[idx].tolist()] for idx in sort_idx if thresh_idx[idx]]
            else:
                data = [{self.data[idx]['code']: rez[idx].item()} for idx in sort_idx]
                if debug:
                    debug_data = [[self.data[idx]['code'], self.data[idx]['text'], rez[idx].item(), self.embs[idx].tolist()] for idx in sort_idx]

            if debug:
                return {'data': data, 'text': text, 'embs': search_emb[0].tolist(), 'debug_data': debug_data}
            else:
                return {'data': data, 'text': text}

    def config(self, data):
        with open('./data/fasttext_cc_config.json', "w", encoding='UTF-8') as write_file:
            json.dump(data, write_file)

        # ToDo Переделать в асинхронный режим
        self.__init__()

    def model_info(self, data):
        info = dict()

        if 'Информация' in data:
            info['модель создана'] = self.model_init
            info['запущена на'] = self.device
            info['время запуска'] = self.startup_time

            if self.model_init:
                info['дата создания'] = self.date_init
                info['эмбеддинги загружены'] = self.embs_is_load

            if self.embs_is_load:
                info['дата загрузки эмбеддингов'] = self.date_load_embs
                info['Время загрузки данных'] = self.time_load_embs

            info['идет обучение'] = self.training
            info['дата запуска обучения'] = self.date_training
            info['время обучения'] = self.fit_time

        if 'Конфигурация' in data:
            if os.path.isfile('./data/fasttext_cc_config.json'):
                with open('./data/fasttext_cc_config.json', "r", encoding='UTF-8') as read_file:
                    info['конфигурационный файл'] = json.load(read_file)
            else:
                info['конфигурационный файл'] = 'отсутствует'

        if 'Данные' in data:
            if self.training:
                info['данные'] = 'сейчас идет обучение модели, запросите данные позже'
            elif os.path.isfile('./data/fasttext_cc_embs.json'):
                with open('./data/fasttext_cc_embs.json', "r", encoding='UTF-8') as read_file:
                    info['данные'] = json.load(read_file)
            else:
                info['данные'] = 'данных нет'

        return info
