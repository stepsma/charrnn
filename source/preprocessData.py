from nltk.corpus import state_union;
from nltk.tokenize import word_tokenize, sent_tokenize;
import numpy as np;
import copy;
import pickle;

corpus_list = ['2001-GWBush-1.txt', '2001-GWBush-2.txt', '2002-GWBush.txt', '2003-GWBush.txt', '2004-GWBush.txt', '2005-GWBush.txt'];
text = [];

for corpus in corpus_list:
    sample_text = state_union.raw(corpus);
    text += sample_text;

print(len(text));

vocabulary = set(text);
char_to_int = {c:i for i,c in enumerate(vocabulary)};
int_to_char = dict(enumerate(vocabulary));
encoded_array = np.array([char_to_int[c] for c in text], dtype=np.int32);

print(len(vocabulary));
print(len(char_to_int));

with open('char_to_int.pickle', 'wb') as f:
    pickle.dump(char_to_int, f);

with open('int_to_char.pickle', 'wb') as f:
    pickle.dump(int_to_char, f);

with open('encoded_array.pickle', 'wb') as f:
    pickle.dump(encoded_array, f);
