import collections
import matplotlib.pyplot as plt
import random
import math
import pickle
from datetime import datetime
import spacy

import os


def carregar(pasta):
    caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
    lista = []
    for i in caminhos:
        review = open(i, 'r')
        lista.append(review.read())
    return lista

def carregar1(arq):
    lista = []
    arq = open(arq, 'r')
    for i in arq:
      lista.append(i)
    return lista
def ler(endereco):
    arq = open(endereco, 'rb')
    return pickle.load(arq)


def salvar(endereco, lista):
    arq = open(endereco, 'wb')
    pickle.dump(lista, arq)


def preprocessamento(arq_carregado, label): ##Lê uma pasta com os arquivos de texto e retorna uma lista dos tokens com a classificação na última posição
    #lista = carregar(pasta) #Carrega todos os arquivos de uma pasta para uma lista onde cada elemento é um texto
    nlp = spacy.load('en')
    lista = arq_carregado
    for n in range(0,len(lista)):

        lista[n] = preprocessamentoS(lista[n], nlp)
        lista[n].append(label)
    return lista

def preprocessamentoS(string, nlp):
    string = nlp(string)

    string = [token.lemma_ for token in string if (not token.is_stop) and ((not token.is_punct or token.text == '!') and  token.is_alpha)]
    return string


#RETORNA O VOCÁBULARIO DAS PALAVRAS MAIS FREQUENTES PARA AS MENOS FREQUENTES
def features(lista, k, x):# lista são todos os documentos ou seja é a concatenação da lista pos e neg , k = quantidade de elementos que eu quero pegar , x apartir de quanto eu quero porque os primeiros não são tão úteis
    dict = {}
    for i in lista:
        for n in i:
            if(n is not 0 and n is not 1): #pego cada token do texto menos o token de classificação
                if n not in dict.keys(): #Checa se não existe a palavra no dicionário
                    dict.update({n: 1})#se não existe adiciono ele no dicionário com valor 1
                else:
                    dict[n] = dict[n] + 1 #se existir incremento a quantidade do que tá la

    dict = sorted(dict,key=dict.get, reverse=True) # ordeno o dicionário decrescente

    if(k is not None and x is not None):
        return (dict[x:x+k]) #Para pegar o k elementos que mais aparecem apartir de x
    else: return dict


#RETORNA TODAS AS PALAVRAS EM UMA SÓ LISTA
def bag_words(docs):
    bag = {0:[], 1:[]}
    for i in docs:
        if i[-1] is 0:
            bag[0] = bag[0]+i
        else: bag[1] = bag[1]+i
    return bag

def contaClasses(docs, classe):
    c = 0
    for i in docs:
        if i[-1] is classe:
            c = c+1
    return c

#CONTA EM QUANTOS DOCUMENTOS DA CLASSE A PALAVRA APARECE
def conta(palavra, documentos, classe):
    soma = 0
    b= False
    if classe is None:
        b = True
    for i in documentos:
        if b or i[-1] is classe:
            if palavra in i:
                soma = soma + 1

    return soma
def contaPalavrasClasse(docs):
    qt = {0:0, 1:0}
    for i in docs:
        qt[i[-1]] = qt[i[-1]]+ len(i)#quantidades de palavras nesse documento
    return qt

def delta(features, bag, docs):
    peso = {}
    for i in features:
        idf = math.log((conta(i, docs, 0) + 1) / (conta(i, docs, 1) + 1))
        peso.update({i:[]})

        for n in docs:
            peso[i].append(n.count(i) * idf)

        peso[i] = sum(peso[i])/len(peso[i])

    return peso

def vetorTF(features, docs):
    vetor = {0:{}, 1:{}}
    print('TF')
    for i in features:
        vetor[0].update({i:[]})
        vetor[1].update({i:[]})
        for n in docs:
             vetor[n[-1]][i].append(n.count(i))
    print('FIM')
    return vetor

def vetorTFIDF(features, docs):
    print('IDF')
    vetor = {0:{}, 1:{}}
    t = len(docs)
    for i in features:
        idf = t/conta(i, docs,None)
        vetor[0].update({i:[]})
        vetor[1].update({i:[]})
        for n in docs:
             vetor[n[-1]][i].append(n.count(i)*math.log(idf))
    print('fim')
    return vetor


def vetorBinario(features,docs):
    print('bin')
    vetor = {0: {}, 1: {}}
    for i in features:
        vetor[0].update({i: []})
        vetor[1].update({i: []})
        for n in docs:
            if i in n:
                vetor[n[-1]][i].append(1)
            else:
                vetor[n[-1]][i].append(0)
    print('fim')
    return vetor

def tf_idf(features, bag, docs):
    peso = {0:{}, 1:{}}
    tam = len(docs)
    for i in features:

        t =  conta(i, docs, None)

        idf = tam/t
        peso[0].update({i:(bag[0].count(i) * math.log( idf))})
        peso[1].update({i:(bag[1].count(i) * math.log( idf))})
    return peso

def binario(features, bag, docs):
    peso = {0:{}, 1:{}}
    for i in features:
        peso[0].update({i:(conta(i, docs, 0))})
        peso[1].update({i:(conta(i, docs, 1))})
def prob(tipo , features, bag, docs):

    if tipo is 1:
        peso = tf_idf(features, bag, docs)
    elif tipo is 2:
        peso = delta(features, bag, docs)
    elif tipo is 3: #binario
        peso = {}
        for i in features:
            peso.update({i:1})

    prob = {0: {}, 1: {}}
    for i in features:
        probabilidade =  (peso+1)/(len(bag[0])+len(features))
        probabilidade1= (peso+1)/(len(bag[1])+len(features))

        prob[0].update({i: math.log(probabilidade*peso)})
        prob[1].update({i: math.log(probabilidade1*peso)})

    return prob

def probabilidade(features, docs, tipo):
    print('começou')
    if tipo is 1:
        peso = vetorTF(features, docs)
    elif tipo is 2:
        peso = vetorTFIDF(features, docs)
    elif tipo is 3: #binario
        peso = vetorBinario(features, docs)

    prob = {0: {}, 1: {}}
    for i in features:
        soma  = 0
        soma1 = 0
        #print (peso[0])

        soma = sum([sum(w) for w in peso[0].values()])

        soma1 = sum([sum(w) for w in peso[1].values()])
        p = (sum(peso[0][i])+1)
        p1 = (sum(peso[1][i])+1)

        probabilidade = p / (soma + len(peso[0]))
        probabilidade1= p1/ (soma1 + len(peso[1]))

        prob[0].update({i: math.log(probabilidade)})
        prob[1].update({i: math.log(probabilidade1)})
    return prob