import pandas as pd
import numpy as np
from iob_transformer import iob_transformer
import matplotlib.pyplot as plt
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


np.random.seed(0)
plt.style.use("ggplot")

''' Leitura do arquivo '''
df = pd.read_csv('docs/contratos_idato.csv')
df_columns = df.columns.tolist()
# ['id_ato', 'id_dodf', 'tipo_rel', 'id_rel', 'anotador_rel', 'timestamp_rel', 'tipo_ent', 'id_ent', 'anotador_ent', 'timestamp_ent', 'offset', 'length', 'texto']




''' Extraindo IOBs do conjunto de dados '''
# Juntando as colunas id_dodf e id_rel
df['id_ato'] = df['id_dodf'] + '-' + df['id_rel']

# Usando a classe iob_transformer fornecida pelo Matheus Stauffer para tratar os dados
iob = iob_transformer('id_ato', 'texto', 'tipo_ent', keep_punctuation=True, return_df=False)

# Vai retornar uma lista de atos e de label
atos, labels = iob.transform(df)




''' Listas para palavras e tags únicas '''

# Palavras
words = set()
for ato in atos:
    for termo in ato:
        words.add(termo)
words = list(words)
words.append("ENDPAD")
words.append("UNK")
num_words = len(words)
print('Número de Termos: ' + str(num_words))


# Tags
tags = set()
for label in labels:
    for tag in label:
        tags.add(tag)
tags = list(tags)
num_tags = len(tags)

print('Número de Tags: ' + str(num_tags))




''' Mapeamento de palavras e tags em valores inteiros '''

# Associando um número inteiro a uma palavra
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}




''' Verifcando a cobertura das sentenças '''

# Usando a função pad_sequences do Keras é possível observar a dispersão
plt.hist([len(ato) for ato in atos], bins=50)
plt.savefig("grafico_tag.png",dpi=300)

max_len = 100
# O max_len é igual 100 devido as características dos dados

# Geração dos pad_sequences e Teste de validação (representção do texto)
X = [[word2idx[w] for w in ato] for ato in atos]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2idx['ENDPAD'])

y = [[tag2idx[w] for w in label] for label in labels]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

# 80% treino, 20% teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# 10% teste, 10% validação
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)




'''
 Tdos os passos anteriores foram necessários para ajustar os dados
'''



''' Rede Neural: Bi-LSTM '''

input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words+1, output_dim=100, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)

model.summary()


# Compilando o modelo de acorco com otimizador, ma função de perda e uma métrica
#  Com a accuracy é possível identificar a qualidade do modelo, poies esse métria é acertiva com as tags vazias
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



''' Com o modelo pronto, agora é parte de treinamento '''
