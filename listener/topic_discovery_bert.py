import umap
import hdbscan
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from listener.config import conf

class TopicDiscoveryBert:

    def __init__(self, data):
        self._data = data
        self._prepared_data = None
        self._model = SentenceTransformer(conf['DEFAULT']['BERT_MODEL_LOCAL_PATH'])
        self._cluster = None
        self._embeddings = None

    def get_data(self):
        #assert data is in a particular format
        if self._prepared_data is None:
            self._data.RESPONSE = self._data.RESPONSE.astype(str)
            self._data['clean_comment'] = self._data['RESPONSE'].apply(lambda x: x.lower())
            self._prepared_data = self._data

    def get_embeddings(self):
        self.get_data()
        self._embeddings = self._model.encode(
                               self._prepared_data['clean_comment']
                              ,show_progress_bar=True)

    def reduce_dims(self, n_components: int=2):
        if self._embeddings is None:
            self.get_embeddings()
        return umap.UMAP(n_neighbors=3, 
                         n_components=n_components, 
                         metric='cosine').fit_transform(self._embeddings)

    def cluster_dims(self):
        self._cluster = hdbscan.HDBSCAN(min_cluster_size=3
                                        ).fit(self.reduce_dims())

    def get_cluster_labels(self):
        if self._cluster is None:
            self.cluster_dims()
        return self._cluster.labels_

    def plot_clustering(self):
        # reduce data to 2 dimensions for plotting
        x = self.reduce_dims()
        result = pd.DataFrame(x, columns=['x', 'y'])
        result['labels'] = self.get_cluster_labels()
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()
        return plt.show()

    def get_doc_dataframe(self):
        self.get_cluster_labels()
        self._prepared_data['Topic'] = self._cluster.labels_
        self._prepared_data['Doc_ID'] = range(len(self._prepared_data))
        self._prepared_data['Doc'] = self._prepared_data['clean_comment']
        return self._prepared_data.groupby(['Topic'], as_index = False) \
                                  .agg({'Doc': ' '.join})

    def get_c_tf_idf(self, documents, m, ngram_range=(1, 1)):
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        return tf_idf, count

    def get_top_n_words_per_topic(self, tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def get_topic_sizes(self, df):
        topic_sizes = (df.groupby(['Topic'])
                         .Doc
                         .count()
                         .reset_index()
                         .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                         .sort_values("Size", ascending=False))
        return topic_sizes

    def get_topic_summary(self):
        docs_per_topic = self.get_doc_dataframe()
        tf_idf, count = self.get_c_tf_idf(docs_per_topic.Doc.values, m=len(docs_per_topic))
        top_n_words = self.get_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
        topic_sizes = self.get_topic_sizes(self._prepared_data)
        return top_n_words, topic_sizes

if __name__=='__main__':
    import pandas
    from listener.config import conf
    df = pandas.read_csv(conf['DEFAULT']['SOURCE_FILE_PATH'])
    TD = TopicDiscoveryBert(data=df)