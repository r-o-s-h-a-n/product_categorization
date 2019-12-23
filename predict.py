import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as RandFor
from sklearn.neural_network import MLPClassifier

# from keras import models
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam

TRAIN_FN = 'train.csv'
LABEL_FN = 'labels.csv'
PREDICT_FN = 'predict.csv'




with open(TRAIN_FN, 'rU') as f:
    freader = csv.DictReader(f)
    brands = sorted(set([x['BrandName'] for x in freader]))
    print('There are {} number of unique brands in training set.'.format(len(brands)))


def onehotify_brand(example_brands):
    out = []
    for b in example_brands:
        v = np.zeros(len(brands))
        if b in brands:
            v[brands.index(b)] = 1
        out.append(v)
    return sp.csr_matrix(out)

def top_n_accuracy(predictions, trues, n):
    best_n = np.argsort(predictions, axis=1)[:,-n:]
    ts = np.argmax(trues, axis=1)
    successes = 0
    for i in range(ts.shape[0]):
      if ts[i] in best_n[i,:]:
        successes += 1
    return float(successes)/ts.shape[0]



TOPK = 2000

def get_ngram_embeddings(train_texts, train_labels, val_texts):
    vectorizer = TfidfVectorizer(ngram_range = (1, 2), # look at unigrams and bigrams
                                strip_accents = 'unicode',
                                decode_error = 'replace',
                                analyzer = 'word',
                                min_df = 2)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOPK, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val




X_title = []
X_brand = []
Y = []
n_examples = 0        

with open('train.csv', 'r') as f:
    freader = csv.DictReader(f)
    for line in freader:
        X_title.append(process_text(line['Title']))
        X_brand.append(line['BrandName'])
        Y.append(line['CategoryName'])
        n_examples += 1
             
N_PARTITIONS = 5
SHUFFLE = True

width = n_examples//N_PARTITIONS

if SHUFFLE:
    random.shuffle(list(zip(X_title, X_brand, Y)))

scores = {'nb':[], 'mlp':[]}
topk = {'nb':[], 'mlp':[]}

for i in range(N_PARTITIONS):
    print('working on partition {}'.format(i))

    X_title_train = X_title[:i*width] + X_title[(i+1)*width:]
    X_brand_train = X_brand[:i*width] + X_brand[(i+1)*width:]
    Y_train = Y[:i*width] + Y[(i+1)*width:]

    X_title_val = X_title[i*width:(i+1)*width]
    X_brand_val = X_brand[i*width:(i+1)*width]
    Y_val = Y[i*width:(i+1)*width]
    
    X_title_train, X_title_val = get_ngram_embeddings(X_title_train, Y_train, X_title_val)
    X_brand_train = onehotify_brand(X_brand_train)
    X_brand_val = onehotify_brand(X_brand_val)
    
    X_train = sp.hstack([X_title_train, X_brand_train])
    X_val = sp.hstack([X_title_val, X_brand_val])
    
    NBModel = MultinomialNB()
    NBModel.fit(X_train, Y_train)
    scores['nb'].append(NBModel.score(X_val, Y_val))
    
    Y_pred = NBModel.predict_proba(X_val)
    Y_true = [[1 if y==j else 0 for j in NBModel.classes_] for y in Y_val]
    topk['nb'].append(top_n_accuracy(Y_pred, Y_true, n=3))
    
    MLPModel = MLPClassifier(activation='relu', alpha=1e-05, batch_size=128,
                hidden_layer_sizes=(64, 64, 64),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=500, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.0, verbose=False, warm_start=False)
    MLPModel.fit(X_train, Y_train)
    scores['mlp'].append(MLPModel.score(X_val, Y_val))

    Y_pred = MLPModel.predict_proba(X_val)
    Y_true = [[1 if y==j else 0 for j in MLPModel.classes_] for y in Y_val]
    topk['mlp'].append(top_n_accuracy(Y_pred, Y_true, n=3))
    
print('\nACCURACY')
print('Partition validation accuracy are: ', scores)
for model in scores:
    print('Mean partition validation score for {}: {}'.format(model.upper(), np.mean(scores[model])))

print('\nTOP 3 ACCURACY')
print('Partition validation accuracy are: ', topk)
for model in  topk:
    print('Mean partition validation score for {}: {}'.format(model.upper(), np.mean(topk[model])))






with open('predict.csv', 'r') as f:
    freader = csv.reader(f)
    
    for row in freader:
        process_text(row[2])
        onehotify_brand(row[1])
        
X_title = []
X_brand = []

with open('predict.csv', 'r') as f:
    freader = csv.DictReader(f)
    for line in enumerate(freader):
        X_title.append(process_text(line['Title']))
        X_brand.append(line['BrandName'])
        
X_title = vectorizer.transform(X_title)
X_title = selector.transform(X_title).astype('float32')
X_brand = onehotify_brand(X_brand)

X = sp.hstack([X_title, X_brand])
Y = NBModel.predict(X)

g = open('predict_out.csv', 'w')
gwriter = csv.writer(g)

with open('predict.csv', 'r') as f:
    freader = csv.reader(f)
    gwriter.writerow(freader.next())
    
    for line, y in zip(freader):
        gwriter.writerow(line.append(y))

g.close()