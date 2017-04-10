# Load the model that we created in Part 2
#collecting bag of words -predicting the word given its context
#skip gram - predicting the context given a word
#
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import time
model = Word2Vec.load("300features_40minwords_10context")
# print(type(model.syn0))
# print(model.syn0.shape)
# print(model["food"])

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] // 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print ("Time taken for K Means clustering: ", elapsed, "seconds.")
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.index2word, idx ))

# For the first 10 clusters
for cluster in range(0,10):
    #
    # Print the cluster number
    print ("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for key, item in word_centroid_map.values():
        if( item == cluster ):
            words.append(item)
    print (words)
