import umap
import hdbscan
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
        self._data.RESPONSE = self._data.RESPONSE.astype(str)
        self._data['clean_comment'] = self._data['RESPONSE'].apply(lambda x: x.lower())
        self._prepared_data = self._data
        return self._prepared_data

    def get_embeddings(self):
        if self._prepared_data is None:
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

if __name__=='__main__':
    import pandas
    from listener.config import conf
    df = pandas.read_csv(conf['COMMPANEL']['SOURCE_FILE_PATH'])
    TD = TopicDiscoveryBert(data=df)