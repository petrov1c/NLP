from flask import jsonify

import torch
from transformers import AutoTokenizer, AutoModel

def cosine_simularity(text1, text2):
    emb1 = embed_bert_cls(text1).unsqueeze(0);
    emb2 = embed_bert_cls(text2).unsqueeze(0);

    return torch.cosine_similarity(emb1, emb2)

def embed_bert_cls(data, return_json = False):

    t = tokenizer(data['СтрокаПоиска'], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)

    if return_json:
        return jsonify({'embedding': embeddings[0].cpu().numpy().tolist()})
    else:
        return embeddings[0]

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)