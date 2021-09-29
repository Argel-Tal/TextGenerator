from COSC343WordEnc import COSC343WordEnc
from COSC343SRNClassifier import COSC343SRNClassifier
import pickle , gzip

"""
selecting text source:
to select a text, modifiy <selection> to one of the following options:
    0 = Karl Marx
    1 = Spam email collection
"""
selection = 1

# hyper params
batchN = 100
lr = 0.001
hiddenSize = 100
epochs = 50
# source text files
textOptions = ["InputTexts/Marx.txt", "InputTexts/SpamEmail.txt"]

# input file opening 
with open(textOptions[selection], "r") as f:
    all_text = f.read()

# data set up as one-hot-encoding
word_enc = COSC343WordEnc(max_text_length=100000)
X = word_enc.fit_transform(all_text)

# training of model
print(str(epochs) + " epochs")
textClass = COSC343SRNClassifier(hidden_layer_size = hiddenSize, 
    activation = 'relu', solver='adam', batch_size = batchN, 
    learning_rate_init = lr, max_iter=epochs, verbose=True, 
    beta_1=0.9, beta_2=0.999, epsilon=1e-08)
textClass.fit(X[:-1], X[1:])

# saving model file
with gzip.open("srn\_SPAMtraining.save", 'w') as f:
    pickle.dump((textClass, word_enc), f)
