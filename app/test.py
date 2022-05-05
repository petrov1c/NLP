from gensim.models import FastText
import numpy as np
from gensim.models import KeyedVectors
import re
import json

s = json.dump(True)
regex = re.compile(r'[A-z]+')

'''
конвертация модели. Внимание, нужна gensim 4.
В колабе используется 3.*

FT = FastText.load_fasttext_format('d:/cc.ru.300.bin')  # Загрузка с моего гугл диска
kv = FT.wv

kv.vectors = np.float16(kv.vectors)
kv.vectors_ngrams = np.float16(kv.vectors_ngrams)
kv.vectors_vocab  = np.float16(kv.vectors_vocab)
kv.save('D:/www/NLP/data/ft.ru.300.float16')
'''

model = KeyedVectors.load('D:/www/NLP/data/ft.ru.300.float16')

# 'фр-7' in model.key_to_index так правильно проверять что слово есть в словаре FastText
idx = []
for word in model.key_to_index:
    rez = regex.findall(word)
    if len(rez) == 0:
        idx.append(model.key_to_index[word])

rez = len(idx)
# так можно искать фразы
# phrases = Phrases(corpus)
# frozen_phrases = phrases.freeze()
