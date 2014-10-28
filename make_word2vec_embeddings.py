from gensim.models import word2vec
import proc
import numpy as np

train_captions = 'data/iaprtc12/train_captions.txt'
context = 5

# Load captions
print 'Loading captions...'
train = proc.load_captions(train_captions)

# Tokenize the data
print 'Tokenizing...'
train_tokens = proc.tokenize(train, context=context)

# Index words and create vocabulary
print 'Creating vocabulary...'
(word_dict, index_dict) = proc.index_words(train_tokens)

model = word2vec.Word2Vec.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

with open('embeddings_word2vec.txt', 'w') as file:
    for word in word_dict:
        if word in model:
            file.write(word + ' ')
            np.savetxt(file, model[word][None], delimiter=" ", fmt='%.6g')
    
    
