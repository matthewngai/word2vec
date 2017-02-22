# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
# nltk.downloads()
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
#print(train.shape)
#print(train.columns.values)
# print (train["review"][0])
#print (train["sentiment"][0])

example1 = BeautifulSoup(train["review"][0])
#print(example1.get_text())


# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
# print(letters_only)
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words

words = [w for w in words if not w in stopwords.words("english")]
#print(words)
#
#porter stemming and lemmatizing

def review_to_words(raw_review):
	review_txt = BeautifulSoup(raw_review).get_text()
	letters_only = re.sub("[^a-zA-Z]", " ", review_txt)
	words = letters_only.lower().split()
	stops = set(stopwords.words("english")) #faster than list
	meaningful_words = [w for w in words if not w in stops]
	return (" ".join(meaningful_words))


#clean_review = review_to_words(train["review"][0])
#print(clean_review)


num_reviews = train["review"].size

print("Cleaning and parsing the training set movie reviews...\n")
clean_trian_reviews = []
for i in range(0, num_reviews):
 	if( (i+1)%1000 == 0 ):
 		print("Review %d of %d\n" % (i+1, num_reviews))
 	clean_trian_reviews.append(review_to_words(train["review"][i]))

print ("Creating the bag of words...\n")
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.

train_data_features = vectorizer.fit_transform(clean_trian_reviews)
train_data_features = train_data_features.toarray()
print(train_data_features.shape)

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
# print(vocab)
#
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print(count, tag)

print( "Training the random forest...")

#100 trees
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["sentiment"] )


test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )


print(test.shape)


num_reviews = len(test["review"])
clean_test_reviews = []

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )


test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

#Forest takes trained data that has sentiment
#then predicts sentiment field in result


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return (featureVec)

    def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return (reviewFeatureVecs)

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print ("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )
