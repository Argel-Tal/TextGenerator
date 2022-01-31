__author__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"

from sklearn import preprocessing
import numpy as np

class COSC343WordEnc:

    def __init__(self,max_text_length=None, max_words=None):

        self.max_text_length = max_text_length
        self.max_words = max_words
        self.enc = None

    def fit(self,X,y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self,X,y=None):

        self.dictionary = list()
        word_indexes = list()

        X = X.replace("--", " ")

        if isinstance(X,str):
            X = " ".join(X.split("\n"))

        n = 0
        tokens = self.split_(line=X)
        for token in tokens:
            try:
                i = self.dictionary.index(token)
            except ValueError:
                i = len(self.dictionary)
                self.dictionary.append(token)
            word_indexes.append(i)
            n+= 1
            if self.max_text_length is not None and n >= self.max_text_length:
                break

            if self.max_words is not None and len(self.dictionary) >= self.max_words:
                break

        return self.indexes_to_one_hot_(word_indexes)

    def transform(self, X):
        if isinstance(X,str):
            tokens = self.split_(X)

        word_indexes = list()

        for token in tokens:
            try:
                i = self.dictionary.index(token)
                word_indexes.append(i)
            except ValueError:
                i = -1


        return self.indexes_to_one_hot_(word_indexes)

    def inverse_transform(self, X):
        X = np.argmax(X,axis=1)
        X = np.array(self.dictionary)[X]
        X = " ".join(X)
        # removing space around punctuation chars
        X = X.replace("•", " -")
        X = X.replace(" ,", ",")
        X = X.replace(" -", "-")
        X = X.replace(" .", ".")
        X = X.replace(" ?", "?")
        X = X.replace(" ;", ";")
        X = X.replace(" ' ", "'")
        X = X.replace(" !", "!")
        X = X.replace("\" ", "\"")

        return X

    def indexes_to_one_hot_(self,x):

        if self.enc is None:
            self.enc = preprocessing.OneHotEncoder(categories='auto')
            self.enc.fit(np.expand_dims(np.arange(len(self.dictionary)),axis=1))

        return self.enc.transform(np.expand_dims(x, axis=1)).toarray()


    def split_(self,line):
        tokens = line.split()
        new_tokens = []
        for i,token in enumerate(tokens):

            while True:
                alphanum = False
                ks = 0
                ke = len(token)
                for j in range(len(token)):
                    if str.isalnum(token[j]):
                        if not alphanum:
                            alphanum = True
                            ks=j
                    else:
                        if token[j]=='\'' or token[j]=='-':
                            continue
                        elif alphanum:
                            ke=j
                            if ke==ks+1 and ke<len(token) and token[ke]=='.':
                                ke += 1
                            break
                        else:
                            ks += 1


                for k in range(ks):
                    new_tokens.append(token[k])

                if ke > ks:
                    new_tokens.append(token[ks:ke])

                if ke < len(token):
                    token = token[ke:]
                else:
                    break
                #for k in range(ke,len(token)):
                #    if str.isalpha(token[k]):
                #        print("Yo")
                #    new_tokens.append(token[k])

        return new_tokens



