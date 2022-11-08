'''
Исправление опечаток. Компонента от DeepPavlov
использует классические алгоритмы, опирается на неизвестную мне kenlm

http://docs.deeppavlov.ai/en/master/features/models/spelling_correction.html

Исправление ошибок использует эту языковую модель
* https://kheafield.com/code/kenlm/
* https://github.com/kpu/kenlm

Зависимости kenlm
* `sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev`

'''

from deeppavlov import build_model, configs

CONFIG_PATH = configs.spelling_correction.levenshtein_corrector_ru

model = build_model(CONFIG_PATH, download=True)