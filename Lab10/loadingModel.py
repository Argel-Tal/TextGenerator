import pickle, gzip
from COSC343WordEnc import COSC343WordEnc

# set up file for write out of generated text
with gzip.open("srn\_SPAMtraining.save") as f:
    marxClass, word_enc = pickle.load(f)

# prompt set up
primer = "the money"
allText = ""
marxClass.reset()

# data into and out of generative model pt1
primerEncoded = word_enc.transform(primer)
output = marxClass.predict(primerEncoded)
decoded = word_enc.inverse_transform(output)
# output management
print(decoded)
allText = primer + " " + decoded

# data into and out of generative model pt2 and output management
for i in range(50):
    primer = decoded
    primerEncoded = word_enc.transform(primer)
    output = marxClass.predict(primerEncoded)
    decoded = word_enc.inverse_transform(output)
    print(decoded)
    allText = allText + " " + decoded

# write out of generated text 
with open("SpamText4.txt", 'w') as g:
    g.write(allText)
    g.close()