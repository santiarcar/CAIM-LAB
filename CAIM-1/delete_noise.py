import string
import os
import time


class noise_cleaner:

    cwd = os.getcwd()

    def __init__(self, search_on):
        self.search_on = search_on
    @staticmethod
    def has_number(word):
        for i in range(10):
            if str(i) in word:
                return True
        return False

    @staticmethod
    def has_punctuation(word):
        for c in word:
            if c in string.punctuation:
                return True
        return False

    @staticmethod
    def not_alnum(word):
        return any(not c.isalnum() for c in word)

    def clean_noise(self):
        f = open('{0}_output.txt'.format(self.cwd + '/' + self.search_on), 'r')
        newOutput = open('{0}_clean.txt'.format(self.cwd + '/' + self.search_on), 'w')
        deprecatedOutput = open('{0}_deprecated.txt'.format(self.cwd + '/' + self.search_on), 'w')

        for linea in f:
            try:
                count = linea.split(sep=', ')[0]
                word = linea.split(sep=', ')[1].strip()
                if not ((word.isdigit()) or (self.has_number(word)) or (self.has_punctuation(word))
                        or self.not_alnum(word) or ('ª' in word) or ('º' in word) or len(word) < 3):
                    newOutput.write(linea)
                else:
                    word = word.replace("_", "")
                    word = word.split("'", 1)[0]
                    if not ((word.isdigit()) or (self.has_number(word)) or (self.has_punctuation(word))
                            or self.not_alnum(word) or ('ª' in word) or ('º' in word) or len(word) < 3):
                        newOutput.write(linea)
                    else:
                        deprecatedOutput.write(linea)

            except Exception as e:
                break

        f.close()
        newOutput.close()
        deprecatedOutput.close()
