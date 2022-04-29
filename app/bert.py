import json
import torch
import os.path

from transformers import AutoTokenizer, AutoModel
from flask import jsonify

import re

REGEX_ONE_STEP = re.compile('\S*\d\S*')  # удаление слов, где есть цифры
REGEX_TWO_STEP = re.compile('[А-я0-9.,!?ёЁ"]+')  # Оставим только русский текст

class Bert:
    def __init__(self, data, model, tokenizer):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        self.tokenizer = tokenizer
        self.model = model

        if isinstance(data, dict):
            embs = torch.zeros((len(data), model.config.emb_size), device=device)
            embs_dict = dict()
            for idx, key in enumerate(data):
                embs[idx] = torch.Tensor(data[key])
                embs_dict[idx] = key

            self.dict = embs_dict
            self.embs = embs

    def predict(self, data):
        if hasattr(self, 'embs'):
            if 'Количество' in data:
                count = data['Количество']
            else:
                count = 3

            search_emb = self.embed_bert_cls(data['СтрокаПоиска'])
            search_emb = search_emb.tile((self.embs.shape[0],1))

            rez = torch.cosine_similarity(search_emb, self.embs)
            sort_idx = rez.argsort(descending=True)[:count].cpu().tolist()

            rez_data = {self.dict[idx]: rez[idx].item() for idx in sort_idx}
            return jsonify({'result': True, 'data': rez_data})
        else:
            return jsonify({'result': False, 'error': 'Модель не инициализирована'})

    def embed_bert_cls(self, data, return_json=False):

        t = self.tokenizer(data, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)

        if return_json:
            return jsonify({'embedding': embeddings[0].cpu().numpy().tolist()})
        else:
            return embeddings[0]

    def cosine_simularity(self, text1, text2):
        emb1 = self.model.embed_bert_cls(text1).unsqueeze(0);
        emb2 = self.model.embed_bert_cls(text2).unsqueeze(0);

        return torch.cosine_similarity(emb1, emb2)

    def init_embeddings(self, data, save_data=True):

        embs = torch.zeros((len(data), self.model.config.emb_size), device=self.model.device)
        embs_dict = dict()

        for i, good in enumerate(data):
            line = REGEX_ONE_STEP.sub('', good['Наименование'].strip())
            line = ' '.join(REGEX_TWO_STEP.findall(line))

            embs[i] = self.embed_bert_cls(line)
            embs_dict[i] = good['Код']

        self.dict = embs_dict
        self.embs = embs

        # сохранение эмбеддингов и инициализация из них
        if save_data:
            bert_embs = {value: embs[key].tolist() for key, value in embs_dict.items()}

            with open('./data/bert_embs.json', "w") as write_file:
                json.dump(bert_embs, write_file)

        return jsonify({'result': True})

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    if os.path.isfile('./data/bert_embs.json'):
        with open('./data/bert_embs.json', "r") as read_file:
            bert_embs = json.load(read_file)
    else:
        bert_embs = ''

    return Bert(bert_embs, model, tokenizer)

def regex(data):
    # https://habr.com/ru/post/349860/
    # https: //regex101.com/r/aGn8QC/2 удобно использовать для отладки шаблонов

    if 'ТипДанных' in data:
        if data['ТипДанных'] == 'Дата':
            sh = '\d{1,2}[.,/]\d{1,2}[.,/]\d{2,4}(?:[\s]?[г][ода]*[.]?)?|\d{1,2}\s(?:янв|фев|мар|апр|мая|июн|июл|авг|сент|окт|ноя|дек)[а-я]*\s\d{2,4}(?:[\s]?[г][ода]*[.]?)?'
        else:
            return jsonify({'error': 'Тип данных {} не поддерживается'.format(data['ТипДанных'])})

        REGEX = re.compile(sh)
        return jsonify({'result': REGEX.findall(data['Строка'].lower().strip())})
    elif 'Шаблон' in data:
        REGEX = re.compile(data['Шаблон'])
        return jsonify({'result': REGEX.findall(data['Строка'].strip())})
    else:
        line = REGEX_ONE_STEP.sub('', data['Строка'].strip())
        return ' '.join(REGEX_TWO_STEP.findall(line))