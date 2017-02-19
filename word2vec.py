# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd       
train = pd.read_csv("../labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
train.shape
train.columns.values