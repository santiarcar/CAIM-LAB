import string
import os

cwd = os.getcwd()

search_on = 'news'


def has_number(word):
    for i in range(10):
        if str(i) in word:
            return True
    return False


def has_punctuation(word):
    for c in word:
        if c in string.punctuation:
            return True
    return False


def not_alnum(word):
    return any(not c.isalnum() for c in word)


f = open('{0}_output.txt'.format(cwd + '/' + search_on), 'r')
newOutput = open('{0}_clean.txt'.format(cwd + '/' + search_on), 'w')
deprecatedOutput = open('{0}_deprecated.txt'.format(cwd + '/' + search_on), 'w')

for linea in f:
    try:
        count = linea.split(sep=', ')[0]
        word = linea.split(sep=', ')[1].strip()
        if not ((word.isdigit()) or (has_number(word)) or (has_punctuation(word))
                or not_alnum(word) or ('ª' in word) or ('º' in word)):
            newOutput.write(linea)
        else:
            word = word.replace("_", "")
            word = word.split("'", 1)[0]
            if not ((word.isdigit()) or (has_number(word)) or (has_punctuation(word))
                    or not_alnum(word) or ('ª' in word) or ('º' in word)):
                newOutput.write(linea)
            else:
                deprecatedOutput.write(linea)

    except Exception as e:
        break
