import pandas as pd
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
# Download the punkt tokenizer for sentence splitting
import nltk.data

print("import statements...")
#read data from files
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#testing
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )

unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print ("Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size ))

def review_to_wordlist(review, remove_stopwords=False):
   	# Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    review_text = BeautifulSoup(review).get_text() #remove HTML
    review_text = re.sub("[^a-zA-Z]", " ", review_text) #remove non letters
    words = review_text.lower().split() #to lowercase and split
    return(words)

print("before tokenizer...")
#check path from User/.../AppData
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

print("after tokenizer...")
# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
	#1. Use the NLTK tokenizer to split the paragraphs into sentences
	raw_sentences = tokenizer.tokenize(review.strip())
	#2. Loop over each sentence
	sentences = []
	for raw_sentence in raw_sentences:
		#If a sentence is empty, skip it
		if len(raw_sentence) > 0:
			#Otherwise, call review_to_wordlist to get a list of words
			sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

			#return the list of sentences (each sentence is a list of words)
			return sentences

sentences = []
print("Parsing sentences from training set")

for review in train["review"]:
	sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
	sentences += review_to_sentences(review, tokenizer)

# Check how many sentences we have in total - should be around 850,000+
print(len(sentences))
print(sentences[0])
print(sentences[1])

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 6       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match("france england germany berlin".split()))
print(model.doesnt_match("paris berlin london austria".split()))
print(model.most_similar("man"))
print(model.most_similar("queen"))
print(model.most_similar("awful"))
