import json
import torch
import os.path

from transformers import AutoTokenizer, AutoModel
from flask import jsonify

import re

class Bert:
    def __init__(self, data, model, tokenizer):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.re_pipeline = dict()
        self.re_pipeline[0] = re.compile(r'\S*\d\S*')          # удаление слов, где есть цифры
        self.re_pipeline[1] = re.compile(r'[А-я0-9.,!?ёЁ"]+')  # Оставим только русский текст

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

            #ToDo Передавать слова в массиве так как в словаре они перемешиваются и выдаются не в том порядке
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
            line = self.re_pipeline[0].sub('', good['Наименование'].strip())
            line = ' '.join(self.re_pipeline[1].findall(line))

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