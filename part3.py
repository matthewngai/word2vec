# Load the model that we created in Part 2
#collecting bag of words -predicting the word given its context
#skip gram - predicting the context given a word
#

model = Word2Vec.load("300features_40minwords_10context")
print(type(model.syn0))
print(model.syn0.shape)
print(model["food"])
