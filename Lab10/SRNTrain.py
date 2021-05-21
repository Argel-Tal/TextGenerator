from COSC343WordEnc import COSC343WordEnc
from COSC343SRNClassifier import COSC343SRNClassifier
import pickle , gzip

with open("Marx.txt", "r") as f:
    all_text = f.read()

batchN = 100
lr = 0.001
hiddenSize = 100
epochs = 50


word_enc = COSC343WordEnc(max_text_length=100000)
X = word_enc.fit_transform(all_text)

print(str(epochs) + " epochs")
marxClass = COSC343SRNClassifier(hidden_layer_size = hiddenSize, activation = 'relu', solver='adam', batch_size = batchN, learning_rate_init = lr, max_iter=epochs, verbose=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

marxClass.fit(X[:-1], X[1:])


with gzip.open("srn\_training.save", 'w') as f:
    pickle.dump((marxClass, word_enc), f)