# https://habr.com/ru/post/349860/
# https: //regex101.com/r/aGn8QC/2 удобно использовать для отладки шаблонов
import re

#ToDo настроить кеширование шаблонов

def run(data):

    if 'Метод' in data:
        if data['Метод'] in ['findall', 'sub']:
            method = data['Метод']
        else:
            method = 'findall'
    else:
        method = 'findall'

    if 'ТипДанных' in data:
        if data['ТипДанных'] == 'Дата':
            sh = re.compile(r'\d{1,2}[.,/]\d{1,2}[.,/]\d{2,4}(?:[\s]?[г][ода]*[.]?)?|\d{1,2}\s(?:янв|фев|мар|апр|мая|июн|июл|авг|сент|окт|ноя|дек)[а-я]*\s\d{2,4}(?:[\s]?[г][ода]*[.]?)?')
        elif data['ТипДанных'] == 'Телефон':
            sh = re.compile(r'[+]?\d+(?:[\s()-]{1,2}\d+){2,}')
        elif data['ТипДанных'] == 'Почта':
            sh = re.compile(r'\S+@\S+[.]\w{1,3}')
        else:
            return {'error': 'Тип данных {} не поддерживается'.format(data['ТипДанных'])}

        data['Строка'] = data['Строка'].lower()
    elif 'Шаблон' in data:
        sh = re.compile(data['Шаблон'])

    if method == 'findall':
        return {'result': sh.findall(data['Строка'].strip())}
    else:
        if 'ТекстЗамены' in data:
            return {'result': sh.sub(data['ТекстЗамены'], data['Строка'].strip())}
        else:
            return {'result': sh.sub('', data['Строка'].strip())}

def pipeline(config, data):
    '''

    :param config: Словарь содержащий 3 массива шаблонов предобработки, обработки и постобработки данных
                        каждый элемент массива содержит имя фукнции и шаблон
    :param data: текст или массив с текстами
    :return: текст или массив с текстами
    '''

    if isinstance(data, str):
        data = [data]

    if 'предобработка' in config:
        for text in data:
            for instruct in config['Предобработка']:
                pass