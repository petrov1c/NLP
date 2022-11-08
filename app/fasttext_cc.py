import fasttext

import torch

import time
from datetime import datetime
import os.path
import re
import json

class SearchModel():
    def __init__(self):
        start_time = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.re = re.compile(r'[^а-яА-ЯёЁ0-9a-zA-Z @!?,.|/:;\'""*&@#$№%\[\]{}()+\-$]')

        self.model_init = False
        self.embs_is_load = False

        if os.path.isfile('./data/fasttext_cc_config.json'):
            with open('./data/fasttext_cc_config.json', "r", encoding='UTF-8') as read_file:
                config = json.load(read_file)

                if 'model' in config:
                    if os.path.isfile(config['model']):
                        self.model = fasttext.load_model(config['model'])
                        self.model_init = True
                        self.datetime_init = datetime.isoformat(datetime.now())

                    self.embs_is_load = False

                    if os.path.isfile('./data/fasttext_cc_embs.json'):
                        with open('./data/fasttext_cc_embs.json', "r") as read_file:
                            loaded_data = json.load(read_file)

                        self.data = loaded_data['data']
                        self.embs = torch.zeros((len(loaded_data['embs']), self.model.get_dimension())).to(self.device)
                        for idx, key in enumerate(loaded_data['embs']):
                            self.embs[idx] = torch.Tensor(loaded_data['embs'][key]).to(self.device)

                        self.embs_is_load = True
                        self.datetime_load_embs = datetime.isoformat(datetime.now())
                        self.load_embs_time = round(time.time() - start_time, 3)

        self.startup_time = round(time.time() - start_time, 3)

    def preprocess(self, text):
        text = text.lower().strip()
        return self.re.sub('', text)

    def fit(self, X):
        start_time = time.time()
        self.embs_is_load = False

        self.data = []
        for item in X:
            line = self.preprocess(item['text'])
            if line != '':
                self.data.append({'code': item['code'], 'text': line})

        self.embs = torch.zeros((len(self.data), self.model.get_dimension())).to(self.device)
        for idx, value in enumerate(self.data):
            self.embs[idx] = torch.Tensor(self.model.get_sentence_vector(value['text'])).to(self.device)

        self.embs_is_load = True
        self.datetime_load_embs = datetime.isoformat(datetime.now())
        self.load_embs_time = round(time.time() - start_time, 3)

        embs = {data['code']: self.embs[idx].tolist() for idx, data in enumerate(self.data)}
        with open('./data/fasttext_cc_embs.json', "w") as write_file:
            save_data = {'data': self.data, 'embs': embs}
            json.dump(save_data, write_file)

    def predict(self, text, top=3, threshhold=0.5, debug=False):
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
            if threshhold > 0:
                thresh_idx = rez > threshhold
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

        self.__init__()

    def model_info(self):
        info = {}
        info['Модель создана'] = self.model_init
        if hasattr(self, 'model'):
            info['vocab'] = len(self.model.words)
            info['dimentions'] = self.model.get_dimension()
            info['device'] = self.device

        if hasattr(self, 'date_init'):
            info['Дата создания'] = self.datetime_init
        if hasattr(self, 'embs_is_load'):
            info['Эмбеддинги загружены'] = self.embs_is_load
        if hasattr(self, 'datetime_load_embs'):
            info['Дата загрузки эмбеддингов'] = self.datetime_load_embs
        if hasattr(self, 'load_embs_time'):
            info['Время загрузки данных'] = self.load_embs_time
        if hasattr(self, 'startup_time'):
            info['Общее время запуска'] = self.startup_time

        if os.path.isfile('./data/fasttext_cc_config.json'):
            with open('./data/fasttext_cc_config.json', "r", encoding='UTF-8') as read_file:
                info['Конфигурационный файл'] = json.load(read_file)
        else:
            info['Конфигурационный файл'] = 'отсутствует'

        return info

