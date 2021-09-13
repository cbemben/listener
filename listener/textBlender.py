from spacy.vocab import Vocab
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from listener.utils.import_data import import_csv, isolate_comment_col

data = import_csv(is_qualtrics_file=True)
comments = isolate_comment_col(data, [9,19,21,23]) 

vocab = Vocab(strings=["hello","world"])
nlp = English()
tokenizer = Tokenizer(nlp.vocab)