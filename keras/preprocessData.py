from nltk.corpus import state_union;
from nltk.tokenize import word_tokenize, sent_tokenize;
import numpy as np;
import copy;
import pickle;
import codecs;
import os;


corpus_files = ['1993-Clinton.txt', '1994-Clinton.txt','1995-Clinton.txt','1996-Clinton.txt','1997-Clinton.txt',
                '1998-Clinton.txt','2000-Clinton.txt'];

##corpus_files = ['1993-Clinton.txt'];
corpus_input = [];

for file in corpus_files:
    file_path = os.path.join("state_union", file);
    print(file_path);
    with codecs.open(file_path, 'r', 'utf-8') as f:
        text = f.read();
        text = text.lower();
        corpus_input += text;

print(len(corpus_input));
##
vocab = set(corpus_input);
char_to_int = {c:i for i,c in enumerate(vocab)};
int_to_char = dict(enumerate(vocab));
encoded_array = np.array([char_to_int[c] for c in corpus_input], dtype=np.int32);

##print(len(vocab));
for key, value in char_to_int.items():
    print(key, value);

##
print(len(vocab));
print(len(char_to_int));
print(len(encoded_array));


##with open('char_to_int.pickle', 'wb') as f:
##    pickle.dump(char_to_int, f);
##
##with open('int_to_char.pickle', 'wb') as f:
##    pickle.dump(int_to_char, f);
##
##with open('encoded_array.pickle', 'wb') as f:
##    pickle.dump(encoded_array, f);
