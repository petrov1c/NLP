from gensim.models import KeyedVectors
from razdel import tokenize            # проект Natasha

import json
import os.path
import time
from datetime import datetime
import re
import torch
import numpy as np
from numpy.linalg import norm

REGEX_ONE_STEP = re.compile(r'\S*\d\S*')          # удаление слов, где есть цифры
REGEX_TWO_STEP = re.compile(r'[А-я0-9.,!?ёЁ]+')  # Оставим только слова с русским текстом
REGEX_THREE_STEP = re.compile(r'\b[^A-zАЕЁИОУЫЭЮЯаеёиоуыэюя0-9\s]{2,}\b') # Удаление слов, состоящих только из согласных букв, например мм

class KeyVectored:
    def __init__(self):
        start_time = time.time()
        self.model_init = False
        self.embs_is_load = False
        if os.path.isfile('./data/fasttext_config.json'):
            with open('./data/fasttext_config.json', "r", encoding='UTF-8') as read_file:
                config = json.load(read_file)

            if 'model' in config:
                if os.path.isfile(config['model']):
                    self.model = KeyedVectors.load(config['model'])
                    self.model_init = True
                    self.datetime_init = datetime.isoformat(datetime.now())
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

                    if 'min_n' in config:
                        self.model.min_n = config['min_n']
                    else:
                        self.model.min_n = 3

                    if 'max_n' in config:
                        self.model.max_n = config['max_n']
                    else:
                        self.model.max_n = 5

                    self.embs_is_load = False

                    if os.path.isfile('./data/fasttext_embs.json'):
                        with open('./data/fasttext_embs.json', "r") as read_file:
                            data = json.load(read_file)
                        if isinstance(data, dict):
                            self.embs = torch.zeros((len(data), self.model.vector_size), device=self.device)
                            self.dict = dict()
                            for idx, key in enumerate(data):
                                self.embs[idx] = torch.Tensor(data[key])
                                self.dict[idx] = key

                            self.embs_is_load = True
                            self.datetime_load_embs = datetime.isoformat(datetime.now())

        self.startup_time = time.time() - start_time

    def model_info(self):
        info = {}
        info['Модель создана'] = self.model_init
        if hasattr(self, 'model'):
            info['vocab'] = self.model.vectors.shape[0]
            info['ngram'] = self.model.vectors_ngrams.shape[0]
            info['min_n'] = self.model.min_n
            info['max_n'] = self.model.max_n
            info['device'] = self.device

        if hasattr(self, 'date_init'):
            info['Дата создания'] = self.datetime_init
        if hasattr(self, 'embs_is_load'):
            info['Эмбеддинги загружены'] = self.embs_is_load
        if hasattr(self, 'datetime_load_embs'):
            info['Дата загрузки эмбеддингов'] = self.datetime_load_embs
        if hasattr(self, 'load_embs_time'):
            info['Время обновления данных'] = self.load_embs_time
        if hasattr(self, 'startup_time'):
            info['Время запуска'] = self.startup_time

        if os.path.isfile('./data/fasttext_config.json'):
            with open('./data/fasttext_config.json', "r", encoding='UTF-8') as read_file:
                info['Конфигурационный файл'] = json.load(read_file)
        else:
            info['Конфигурационный файл'] = 'отсутствует'

        return info

    def update(self, data, save_data=True):
        start_time = time.time()

        self.embs_is_load = False

        data_list = []
        for _ in data:
            line = self.preprocess(_['Наименование'])
            if line != '':
                data_list.append({'Код': _['Код'], 'line': line})

        self.embs = torch.zeros((len(data_list), self.model.vector_size), device=self.device)
        self.dict = dict()
        for idx, _ in enumerate(data_list):
            self.embs[idx] = torch.Tensor(self.get_sentence_vector(_['line'])).to(self.device)
            self.dict[idx] = _['Код']

        # сохранение эмбеддингов и инициализация из них
        if save_data:
            embs = {value: self.embs[key].tolist() for key, value in self.dict.items()}
            with open('./data/fasttext_embs.json', "w") as write_file:
                json.dump(embs, write_file)

        self.embs_is_load = True
        self.datetime_load_embs = datetime.isoformat(datetime.now())
        self.load_embs_time = time.time() - start_time

        return {'result': True}

    def config(self, data):
        with open('./data/fasttext_config.json', "w", encoding='UTF-8') as write_file:
            json.dump(data, write_file)

        self.__init__()

    def predict(self, data):
        if self.model_init and self.embs_is_load:
            if 'Количество' in data:
                count = data['Количество']
            else:
                count = 3

            if 'Порог' in data:
                threshhold = data['Порог']
                if threshhold > 1:
                    threshhold = threshhold/100
            else:
                threshhold = 0

            text = self.preprocess(data['СтрокаПоиска'])
            if text == '':
                return {'result': True, 'data': [], 'text': text}

            search_emb = torch.Tensor(self.get_sentence_vector(text)).to(self.device)
            search_emb = search_emb.tile((self.embs.shape[0], 1)).to(self.device)

            rez = torch.cosine_similarity(search_emb, self.embs)
            sort_idx = rez.argsort(descending=True)[:count].tolist()
            if threshhold > 0:
                thresh_idx = rez > threshhold
                rez_data = [{self.dict[idx]: rez[idx].item()} for idx in sort_idx if thresh_idx[idx]]
            else:
                rez_data = [{self.dict[idx]: rez[idx].item()} for idx in sort_idx]
            return {'result': True, 'data': rez_data, 'text': text}

        elif not self.model_init:
            return {'result': False, 'error': 'Создайте модель'}
        else:
            return {'result': False, 'error': 'Загрузите эмбеддинги'}

    def preprocess(self, text):
        '''
        привожу к нижнему регистру
        удаляем слова с цифрами
        оставляем слова только с русским текстом и цифрами
        оставляем слова с русскими гласными буквами
        '''
        text = text.lower().strip()
        text = REGEX_ONE_STEP.sub('', text)
        text = ' '.join(REGEX_TWO_STEP.findall(text))
        text = REGEX_THREE_STEP.sub('', text)
        text = ' '.join(text.split())
        return text

    def get_sentence_vector(self, text):

        words = [i.text for i in tokenize(text)]

        sent_vec = np.zeros(shape=self.model.vector_size)

        num_words = 0
        for word in words:
            try:
                vec = self.model.get_vector(word)
            except:
                continue

            num_words += 1
            vec = vec.copy()
            norm_vec = norm(vec)
            if norm_vec > 0:
                vec /= norm_vec
            sent_vec += vec

        if num_words > 1:
            sent_vec = sent_vec / num_words

        return sent_vec