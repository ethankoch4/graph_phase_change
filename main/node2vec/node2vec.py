
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import sys
sys.path.append('../')
import numpy as np
import networkx as nx
from node2vec.graph import Node2VecGraph
from gensim.models.word2vec import Word2Vec

# personal preference
np.set_printoptions(precision=3, suppress=True)


class node2vec(object):
    def __init__(
            self,
            G=None,
            Adj_M=None,
            labels=[],
            n_classes=None,
            evaluate=False,
            p=1,
            q=1,
            walk_length=50,
            num_walks=25,
            window_size=10,
            embedding_size=128,
            num_iter=4,
            min_count=0,
            sg=1,
            workers=8,
            score=False
            ):
        if G:
            nx_G = G
        else:
            nx_G = self.read_graph(Adj_M)
        G = Node2VecGraph(nx_G, False, p, q)
        G.preprocess_transition_probs()
        self.walks = G.simulate_walks(num_walks, walk_length)
        self.model = self.learn_embeddings(self.walks,
                                           window_size=window_size,
                                           embedding_size=embedding_size,
                                           num_iter=num_iter,
                                           min_count=min_count,
                                           sg=sg,
                                           workers=workers
                                           )
        self.labels = labels
        if evaluate:
            self.kmeans_evaluate(self.model,
                                 labels=labels,
                                 n_clusters=n_classes,
                                 score=score
                                 )

    def read_graph(self, Adj_M):
        # only support undirected graphs as of now
        G = nx.from_numpy_matrix(Adj_M)
        G = G.to_undirected()
        return G

    def learn_embeddings(self, walks, window_size=10, embedding_size=128, num_iter=4, min_count=0, sg=1, workers=8):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [list(map(str, walk)) for walk in walks]
        return Word2Vec(walks, size=embedding_size, window=window_size, min_count=min_count, sg=sg, workers=workers, iter=num_iter)

    def kmeans_evaluate(self, embeddings, labels=[], n_clusters=2, score=False):
        from sklearn.cluster import KMeans
        from utilities import score_bhamidi
        from utilities import score_purity
        from utilities import score_agreement

        if not labels == []:
            walks_data = embeddings.wv
            walks_data = [walks_data[str(i)] for i in range(len(labels))]

            kmeans = KMeans(n_clusters=n_clusters).fit(walks_data)
            self.predicted_labels = list(kmeans.labels_)
            # if scores should be generated
            if score:
                self.bhamidi_score = score_bhamidi(labels, list(kmeans.labels_))
                self.purity_score = score_purity(labels, list(kmeans.labels_))
                self.agreement_score = score_agreement(labels, list(kmeans.labels_))
