#!/usr/bin/env python
# coding: utf-8

# In[3]:


from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json
import nltk
import requests
import re
import pandas as pd
import numpy as np
from keybert import KeyBERT
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer


# # Scrapping:
# Para obtener el corpus que luego permitirá realizar el análisis. Para realizar este escrapeo se ha usado las herramientas de BeautifulSoup, con ella se puede acceder al código HTML de la página web que se quiera y realizar desde ahí la obtención de los datos. En concreto en el siguiente código se accede a los mensajes de un foro sobre ludopatía, para ello se parte de la página donde estan los temas de foros y se van obteniendo los links de cada tema de conversacíon. Una vez obtenidos esos links se recorren y se van extrayendo los mensajes que nos interesen con usuario, fecha y título. Una vez obtenidos los datos se guardan en un dataframe.

# In[4]:


#Declaraciones para beautifull soup
url="https://www.ludopatia.org/forum/"
response = requests.get("https://www.ludopatia.org/forum/forum_topics.asp?FID=1")
soup = BeautifulSoup(response.content, 'html.parser')
used_links = []
next_page=True
counter=0 #contador del número de páginas que quiero recorrer
#Expresiones regulares para el filtrado de los textos
patronResp =r'^[^\s]+ escribió.*$'
patronFecha=re.compile( r"Escrito el: .* [0-9]{2}:[0-9]{2}")
patronTitle=re.compile(r"Tema:.*")
df = pd.DataFrame(columns=['users', 'msg', 'date', 'title'])#dataframe donde se guardará el corpus
#Obtenemos los links de las páginas del foro donde se publican los mensajes
while next_page==True and counter<10:
    links = soup.find_all('a')
    links_repe = soup.select('a[href$="TPN=1"]')
    for link in links:
        link_url = link.get('href')
        if link_url.startswith('forum_posts') and link.find('img') is None and link not in links_repe:
            link_url=url+link_url
            used_links.append(link_url)
    if 'Siguiente' in links[len(links)-3].text:
        link_url = links[len(links)-3].get('href')
        link_url=url+link_url
        response=requests.get(link_url)
        soup = BeautifulSoup(response.content, 'html.parser')
    else:
        next_page=False
    counter=counter+1

#Recorremos los links y guardamos los mensajes candidatos junto al título del post, usuario y fecha
for link in used_links:
    response= requests.get(link)
    soup = BeautifulSoup(response.content)
    texto_title=patronTitle.search(soup.text)[0].split(':')[1]
    users = soup.find_all('span', class_='bold')
    msgs= soup.find_all('td', class_='text')
    i=1
    try:
        for user in users:
            texto_msg = re.sub(r'\bEditado\b\W.*', '', msgs[i].text)
            texto_msg=texto_msg.strip()

            if re.search(patronResp, texto_msg, flags=re.MULTILINE):
                i=i+1
            else:
                if len(texto_msg) > 0:
                    texto_user=user.text
                    texto_date = patronFecha.search(texto_msg)[0].split(':')[1]
                    texto_msgf = texto_msg.split("__________________")[0]
                    texto_msgf= re.split(patronFecha,texto_msgf)[1].strip()
                    texto_msgf = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚ\s]', '',texto_msgf)
                    if len(texto_msgf) > 0  :
                        row= pd.DataFrame({'users': [texto_user], 'msg': [texto_msgf],'date':[texto_date],'title':[texto_title]})
                        df = pd.concat([df, row],ignore_index=True)

            i=i+2
    except:
        i=i+2
        continue
print(df)


# # Guardado del corpus:
# Para guardar el corpus obtenido, se genera un archivo json que guarda todos los datos de manera ordenada. Después podremos leer estes datos y meterlos de nuevo en un dataframe. Como se puede ver el dataframe tiene 3393 mensajes guardados.

# In[5]:


#Guardamos los datos del foro en un archivo json
df.to_json(orient='index', force_ascii=False, path_or_buf=r"C:\Users\jatop\master\Noestructurados\corpus.json")


# In[6]:


#Recuperamos el archivo de json
df = pd.read_json(r"C:\Users\jatop\master\Noestructurados\corpus.json", orient='index',encoding='utf-8')
print(df)


# # TF/IDF:
# El primer análisis se realiza con la ponderacíon tf/idf. En primer lugar queremos conseguir los 50 terminos mas centrales, para ello se crea el CountVectorized y la matriz de terminos de frecuencia , despues se suman las frecuencias de cada término y se sacan los 50 terminos con mayor frecuencia.
# 
# Para sacar los 100 terminos con mayor aparición en el texto el procedimiento es similar pero se crea el countVectorized, el cual nos calculara las palabras con mas aparición en los textos, del mismo modo se suman los resultados de la matriz y se muestran los 100 primeros con la mayor suma.

# In[7]:


#Obtenemos las stopwords en español de nltk
stopwords = nltk.corpus.stopwords.words('spanish')
# Extraer la columna "msg" de el dataframe y almacenarla en una lista
msgs  = df['msg'].tolist() 


# In[8]:


# Crear una instancia de CountVectorizer y configurar los parámetros para eliminar las palabras de menos de 3 letras, stopwords y palabras que aparezcan en menos de 10 documentos
vectorizer =TfidfVectorizer(stop_words=stopwords, min_df=10, token_pattern=r'\b\w{4,}\b') 

# Utilizar la función fit_transform para crear una matriz de términos de frecuencia
X = vectorizer.fit_transform(msgs)
terms = vectorizer.get_feature_names_out()

# Sumar las frecuencias de cada término en la matriz
frequencies = np.asarray(X.sum(axis=0)).ravel()

# Obtener los 50 terminos mas centrales
top_indices = frequencies.argsort()[:-51:-1] 
top_terms = [terms[i] for i in top_indices]

 # Imprimir las 50 palabras más centrales
print(top_terms)


# In[9]:


# Crear una instancia de CountVectorizer y configurar los parámetros para eliminar las palabras de menos de 3 letras, stopwords y palabras que aparezcan en menos de 10 documentos
vectorizer =CountVectorizer(stop_words=stopwords, min_df=10, token_pattern=r'\b\w{4,}\b')

# Utilizar la función "fit_transform" de CountVectorizer para crear una matriz de términos de frecuencia a partir de la lista de textos y la lista de palabras
X = vectorizer.fit_transform(msgs)
palabras = vectorizer.get_feature_names_out() 

# Calcular la suma de tf para cada palabra en la colección
tf_suma = np.asarray(X.sum(axis=0)).ravel() 

# Obtener los índices de las 100 palabras más repetidas
indices_palabras_repetidas = tf_suma.argsort()[:-101:-1] 

#Obtener las palabras correspondientes a los índices de las 100 palabras más repetidas
palabras_repetidas = [palabras[i] for i in indices_palabras_repetidas]

# Imprimir las 100 palabras más repetidas
print("100 términos más repetidos de la colección:")
print(palabras_repetidas) 


# # KeyBert
# Otra forma de realizar el analisis de palabras es mediante la similitud coseno. Para ello se ha utilizado un modelo Bert, primero se carga el modelo deseado, despues se crea el objeto Keybert, despues extraemos las palabras y las vamos añadiendo a un diccionario de forma que si es la primera vez que aparece se inserta y si ya esta en el diccionario se suma su distancia, una vez acabado el cálculo se muestran los términos.

# In[10]:


#Se carga el modelo de KeyBert
modelo = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')


# In[11]:


# Crear objeto KeyBERT
keybert_model = KeyBERT(model=modelo)

# Crear diccionario para almacenar la distancia acumulada de cada palabra clave
keyword_distances = defaultdict(int)

# Crear diccionario para almacenar el número de documentos en los que aparece cada palabra clave
keyword_counts = defaultdict(int)

# Extraer palabras clave de los mensajes y actualizar la distancia acumulada y el número de documentos de cada palabra clave
for msg in msgs:
    keywords = keybert_model.extract_keywords(msg)
    for keyword in keywords:
        if len(keyword[0]) > 2 and keyword[0] not in stopwords:
            keyword_distances[keyword[0]] += keyword[1]
# Filtrar las palabras clave por stopwords y palabras de más de 2 letras 
filtered_keywords = [(k, v) for k, v in keyword_distances.items() if k not in stopwords and len(k) > 2]

# Ordenar las palabras clave por distancia acumulada
filtered_keywords = sorted(filtered_keywords, key=lambda x: x[1], reverse=True)

# Imprimir las palabras clave filtradas
for keyword in filtered_keywords:
    print(keyword[0])


# # WordCloud
# Para el worcloud hemos usado el diccionario creado con anterioridad en el apartado de KeyBert, de forma que las palabras con mayor distancia acumulada son mas grandes.

# In[12]:


wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10)
wordcloud.generate_from_frequencies(keyword_distances)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

