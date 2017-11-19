import nltk, math, codecs
from gensim.models import Doc2Vec
from nltk.cluster.kmeans import KMeansClusterer
import re

from nltk.corpus import stopwords


NUM_CLUSTERS = 20

def preprocess(str):
    # remove links
    str = re.sub(r'http(s)?:\/\/\S*? ', "", str)
    return str


def preprocess_document(text):
    text = preprocess(text)
    return ''.join([x if x.isalnum() or x.isspace() else " " for x in text ]).split()

#data = <sparse matrix that you would normally give to scikit>.toarray()
fname = "trend-index-dataset.model"
model = Doc2Vec.load(fname)

corpus = codecs.open('trend-dataset-small.txt', mode="r", encoding="utf-8")
lines = corpus.read().lower().split("\n")
count = len(lines)

vectors = []

print("inferring vectors")
duplicate_dict = {}
used_lines = []
for i, t in enumerate(lines):
    if i % 2 == 0 and t not in duplicate_dict:
        duplicate_dict[t] = True
        used_lines.append(t)
        vectors.append(model.infer_vector(preprocess_document(t)))

print("done")



kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(vectors, assign_clusters=True)


# clustersizes = []
#
# def distanceToCentroid():
#     for i in range(0,NUM_CLUSTERS):
#         clustersize = 0
#         for j in range(0,len(assigned_clusters)):
#             if (assigned_clusters[j] == i):
#                 clustersize+=1
#         clustersizes.append(clustersize)
#         dist = 0.0
#         centr = means[i]
#         for j in range(0,len(assigned_clusters)):
#             if (assigned_clusters[j] == i):
#                 dist += pow(nltk.cluster.util.cosine_distance(vectors[j], centr),2)/clustersize
#         dist = math.sqrt(dist)
#         print("distance cluster: "+str(i)+" RMSE: "+str(dist)+" clustersize: "+str(clustersize))


def get_titles_by_cluster(id):
    list = []
    for x in range(0, len(assigned_clusters)):
        if (assigned_clusters[x] == id):
            list.append(used_lines[x])
    return list

def get_topics(titles):
    from collections import Counter
    words = [preprocess_document(x) for x in titles]
    words = [word for sublist in words for word in sublist]
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    count = Counter(filtered_words)
    print(count.most_common()[:5])


def cluster_to_topics(id):
    get_topics(get_titles_by_cluster(id))