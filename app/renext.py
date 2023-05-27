__author__ = 'Aleksandr Petrov <petrov1c@yandex.ru>'
# https://habr.com/ru/post/349860/
# https: //regex101.com/r/aGn8QC/2 удобно использовать для отладки шаблонов
import re

#ToDo настроить кеширование шаблонов

def run(data):
    if 'Метод' in data:
        if data['Метод'] in ['findall', 'finditer', 'sub']:
            method = data['Метод']
        else:
            method = 'findall'
    else:
        method = 'findall'

    if 'ТипДанных' in data:
        if data['ТипДанных'] == 'Дата':
            #ToDo Добавить в шаблон время
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
        result = sh.findall(data['Строка'])
        return {'result': result}
    elif method == 'finditer':
        match = sh.finditer(data['Строка'])
        result = [[val.start(), val.end(), val.group()] for val in match]
        return {'result': result}
    else:
        if 'ТекстЗамены' in data:
            result = sh.sub(data['ТекстЗамены'], data['Строка'])
        else:
            result = sh.sub('', data['Строка'])
        return {'result': result}

def pipeline(pipeline, data):
    '''
    :param config: Массив, содержащий шаги обработки
    :param data: текст или массив с текстами
    :return: текст или массив с текстами
    '''

    one_sample = isinstance(data, str)
    if one_sample:
        data = [data]

    result = []
    for text in data:
        text = text.lower()
        for step in pipeline:
            step["Строка"] = text
            answer = run(step)

            if 'error' in answer:
                result.append({'error': answer['error']})
                break
            elif isinstance(answer["result"], str):
                text = answer["result"]
            elif isinstance(answer["result"], list):
                if len(answer["result"]):
                    text = answer["result"][0]
                else:
                    text = ""

            if text == "":
                break
        result.append({'result': text})

    if one_sample:
        return result[0]
    else:
        return result