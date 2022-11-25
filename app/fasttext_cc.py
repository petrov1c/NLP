import fasttext
import torch

import threading  # Для запуска фонового обучения

import os.path
import re
import json
import time
from datetime import datetime


class SearchModel:
    def __init__(self, name):
        start_time = time.time()
        self.name = name
        self.data_file = './data/fasttext/{}.json'.format(name)
        self.config_file = './data/fasttext/{}_config.json'.format(name)
        self.model_init = False
        self.training = False
        self.embs_is_load = False
        self.fit_time = 0
        self.date_training = datetime.isoformat(datetime(1, 1, 1))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.re = re.compile(r'[^а-яА-ЯёЁ0-9a-zA-Z @!?,.|/:;\'""*&@#$№%\[\]{}()+\-$]')

        if os.path.isfile(self.config_file):
            with open(self.config_file, "r", encoding='UTF-8') as read_file:
                config = json.load(read_file)

            if 'model' in config:
                if os.path.isfile('./data/models/{}'.format(config['model'])):
                    if not config['model'] in fasttext_models:
                        fasttext_models[config['model']] = fasttext.load_model('./data/models/{}'.format(config['model']))

                    self.model = fasttext_models[config['model']]
                    self.model_init = True
                    self.date_init = datetime.isoformat(datetime.now())

                if os.path.isfile(self.data_file):
                    with open(self.data_file, "r") as read_file:
                        loaded_data = json.load(read_file)

                    if len(loaded_data['embs'][0])>0:
                        if self.model.get_dimension() == len(loaded_data['embs'][0]):
                            self.data = loaded_data['data']
                            self.embs = torch.Tensor(loaded_data['embs']).to(self.device)
                            self.embs_is_load = True

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
        self.date_training = datetime.isoformat(datetime.now())
        self.fit_time = round(time.time() - start_time, 3)

        with open(self.data_file, "w") as write_file:
            save_data = {'data': self.data, 'embs': self.embs.tolist()}
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
                    debug_data = [
                        [self.data[idx]['code'], self.data[idx]['text'], rez[idx].item(), self.embs[idx].tolist()] for
                        idx in sort_idx if thresh_idx[idx]]
            else:
                data = [{self.data[idx]['code']: rez[idx].item()} for idx in sort_idx]
                if debug:
                    debug_data = [
                        [self.data[idx]['code'], self.data[idx]['text'], rez[idx].item(), self.embs[idx].tolist()] for
                        idx in sort_idx]

            if debug:
                return {'data': data, 'text': text, 'embs': search_emb[0].tolist(), 'debug_data': debug_data}
            else:
                return {'data': data, 'text': text}

    def config(self, data):
        with open(self.config_file, "w", encoding='UTF-8') as write_file:
            json.dump(data, write_file)

        # ToDo Переделать в асинхронный режим
        self.__init__(self.name)

    def model_info(self, data):
        info = dict()

        if 'Информация' in data:
            info['модель создана'] = self.model_init
            info['запущена на'] = self.device
            info['время запуска'] = self.startup_time

            if self.model_init:
                info['дата создания'] = self.date_init
                info['эмбеддинги загружены'] = self.embs_is_load

            info['идет обучение'] = self.training
            info['дата обучения'] = self.date_training
            info['время обучения'] = self.fit_time

        if 'Конфигурация' in data:
            if os.path.isfile(self.config_file):
                with open(self.config_file, "r", encoding='UTF-8') as read_file:
                    info['конфигурационный файл'] = json.load(read_file)
            else:
                info['конфигурационный файл'] = 'отсутствует'

        if 'Данные' in data:
            if self.training:
                info['данные'] = 'сейчас идет обучение модели, запросите данные позже'
            elif os.path.isfile(self.data_file):
                with open(self.data_file, "r", encoding='UTF-8') as read_file:
                    info['данные'] = json.load(read_file)
            else:
                info['данные'] = 'данных нет'

        return info


fasttext_models = dict()
