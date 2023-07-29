import pandas as pd
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
# Download the punkt tokenizer for sentence splitting
import nltk.data

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

tokenizer = nltk.data.load('tokenizer/punkt/english.pickle')
