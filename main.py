import classificador as c
import math
from datetime import datetime
import random

def treino(documentos,tamanho, apartir, tipo):

    feature = c.features(documentos, tamanho, apartir)
    bag = c.bag_words(documentos)

    #prob = c.prob(tipo, feature,bag, documentos)  # se não der certo fazer já no dicionario ta demorando
    prob = c.probabilidade(feature, documentos, tipo)
    priori = {0:0, 1:0}
    soma1 = 0
    soma0 = 0
    for d in documentos:
        if d[-1] is 1:
            soma1 = soma1 + 1
        else :
            soma0 = soma0 + 1

    priori [0] = math.log (soma0/(soma0+soma1))
    priori [1] = math.log (soma1/(soma0+soma1))

    return priori, prob

def teste(priori, logs, testedoc, tipo):
    classe = [priori[0], priori[1]]
    for i in testedoc:    #calcular a exponencial de log

            if i in logs[0].keys():

                classe[0] = classe[0]+(logs[0][i])
                classe[1] = classe[1]+(logs[1][i])

    if classe[0]>classe[1]:
        return 0
    else: return 1

documento = c.ler('doc1.txt')


"""priori, prob = treino(documento[400:2000],2000, 0, 1)
n_acertos = 0
for i in documento[0:400]:
   if teste(priori, prob, i, 3) is i[-1]:           
        n_acertos = n_acertos + 1
print(n_acertos)
"""
print(datetime.now())
priori, prob = treino(documento[400:2000],None,None,2)

print(datetime.now())
n_acertos = 0
for i in documento[0:400]:
   if teste(priori, prob, i, 3) is i[-1]:
        n_acertos = n_acertos + 1
print(n_acertos)
