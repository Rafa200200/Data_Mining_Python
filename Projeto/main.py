import json
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

file = pd.read_json("ethtweets.json")
df = pd.DataFrame(file)

#O Comando "del df", elimina algumas colunas que eu achei desnecessárias à tabela
del df["user_location"]
del df["user_description"]
del df["user_created"]
del df["user_friends"]
del df["user_favourites"]
del df["hashtags"]
del df["source"]
del df["is_retweet"]

df.head() #Imprime os 5 primeiros tweets da list, assim como os seus detalhes, username e etc...´

df.shape #Diz-nos quantas linhas e quantas colunas tem o nosso dataset, respetivamente

df.info() #Algumas infos uteis à cerca da nossa tabela, dados que vão ser úteis no futuro para criar métodos para limpar o texto

duplicados = df[df.duplicated(keep='first')] #Verifica se há dados repetidos
print(duplicados)

#Limpar menções e mudanças de linha desnecessárias
mentions = ["@[A-Za-z0-9]", "\n"]
for char in mentions:
    df['text'] = df['text'].str.replace(char, '')

#10 entidades mais relevantes deste dataset
nlp.max_length = 1030000                       #mudei o máximo aceite pois o padrão "1000000" não chegavaa para o meu dataset inteiro
tokens = nlp(''.join(str(df.text.tolist())))
items = [x.text for x in tokens.ents]
print("="*80)
print("   IDENTIDADES MAIS USADAS NESTES TWEETS E Nº DE VEZES QUE É USADA")
print("="*80)
Counter(items).most_common(10)

#WordCloud
allWords = " ".join( [twts for twts in df['text']] )
wordCloud = WordCloud(width = 500, height = 300, random_state = 10, max_font_size = 119).generate(allWords)

plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

#Criar uma função que nos apresente a subjectividade
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

#Criar uma função que nos apresenta a polaridade
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

#Criar duas colunas na tabela, uma que corresponde aos valores da função "getSubjetivity" e outra que corresponda ao valor da função "getPolarity"
df["Subjectivity"] = df['text'].apply(getSubjectivity)
df["Polarity"] = df['text'].apply(getPolarity)

df

#Criar uma função que a partir do valores obtidosa anteriormente nos indique se o sentimento é Positivo, Neutro ou Negativo
def getAnalise(valor):
  if valor < 0:
    return "Negative"
  elif valor == 0.00:
    return "Neutral"
  else:
    return "Positive"

#Criar mais uma coluna na tabela, que nos indique o valor que a função "getAnalise" retorna
df["Analysis"] = df["Polarity"].apply(getAnalise)

df

#Imprimir todos os tweets positivos PS: O número à esquerda não é o número do tweet no dataset, apenas uma contagem dos tweets imprimidos
print("="*80)
print("             TODOS OS TWEETS POSITIVOS")
print("="*80)

a=1
sortedDF = df.sort_values(by=["Polarity"])
for i in range (0, sortedDF.shape[0]):
  if(sortedDF["Analysis"][i] == "Positive"):
    print(str(a) + ") " + sortedDF['text'][i])
    a += 1

#Imprimir todos os tweets negativos PS: O número à esquerda não é o número do tweet no dataset, apenas uma contagem dos tweets imprimidos
print("="*80)
print("              TODOS OS TWEETS NEGATIVOS")
print("="*80)

a=1
sortedDF = df.sort_values(by=['Polarity'], ascending = 'False')
for i in range (0, sortedDF.shape[0]):
  if(sortedDF["Analysis"][i] == "Negative"):
    print(str(a) + ") " + sortedDF['text'][i])
    a += 1

#Imprimir todos os tweets neutros PS: O número à esquerda não é o número do tweet no dataset, apenas uma contagem dos tweets imprimidos
print("="*80)
print("              TODOS OS TWEETS NEUTROS")
print("="*80)

a=1
sortedDF = df.sort_values(by=['Polarity'], ascending = 'False')
for i in range (0, sortedDF.shape[0]):
  if(sortedDF["Analysis"][i] == "Neutral"):
    print(str(a) + ") " + sortedDF['text'][i])
    a += 1

#Criar um gráfico de análise sentimental com os valores obtidos anteriormente
print("="*80)
print("GRÁFICO DE ANÁLISE SENTIMENTAL TENDO COMO BASE A POLARIDADE E A SUBJECTIVIDADE")
print("="*80)

plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
  plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color="Green")

plt.title("Sentiment Analysis")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()

#Valor em percentagem de tweets positivos

ptweets = df[df.Analysis == "Positive"]
ptweets = ptweets['text']
print("% DE TWEETS POSITIVOS:")
round( (ptweets.shape[0] / df.shape[0]) *100, 1)

#Valor em percentagem de tweets negativos
ntweets = df[df.Analysis == "Negative"]
ntweets = ntweets['text']
print("% DE TWEETS NEGATIVOS:")
round( (ntweets.shape[0] / df.shape[0] *100), 1)

#Valor em percentagem de tweets neutros
neutraltweets = df[df.Analysis == "Neutral"]
neutraltweets = neutraltweets['text']
print("% DE TWEETS NEUTROS:")
round( (neutraltweets.shape[0] / df.shape[0] *100), 1)

#Gráfico de barras final com as conclusões retiradas a partir da análise sentimental feita
print("="*80)
print("GRÁFICO DE BARRAS COM A ANÁLISE SENTIMENTAL CONCLUÍDA")
print("="*80)

df["Analysis"].value_counts()

plt.title("Sentimental Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Counts")
df["Analysis"].value_counts().plot(kind="bar")
plt.show()