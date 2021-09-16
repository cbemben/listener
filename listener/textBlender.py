import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

from listener.utils.import_data import import_csv, isolate_comment_col

data = import_csv(is_qualtrics_file=True)
comments = isolate_comment_col(data, [9,19,21,23]) 

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')
com = comments
com = com.dropna()
com["response_text"] = com.iloc[:,1].apply(lambda x: 'What made scheduling the appointment? ' + x)
com["toks"] = com.iloc[:,1].apply(lambda x: nlp(x))
com["polarity"] = com["toks"].apply(lambda x: x._.polarity)
com["subjectivity"] = com["toks"].apply(lambda x: x._.subjectivity)
com["assessments"] = com["toks"].apply(lambda x: x._.assessments)

from matplotlib import pyplot as plt
plt.hist(com["polarity"])
plt.show()