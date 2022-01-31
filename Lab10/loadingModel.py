import pickle, gzip
from COSC343WordEnc import COSC343WordEnc

"""
set file name for each run
set desired length of text output (words)
"""
desiredFileName = "spamDefinitelyTheFBI - 01"
textLength = 60


# set up file for write out of generated text
with gzip.open("srn\_SPAMtraining.save") as f:
    textClass, word_enc = pickle.load(f)

# prompt set up
primer = "my friend"
allText = ""
textClass.reset()

# data into and out of generative model pt1
primerEncoded = word_enc.transform(primer)
output = textClass.predict(primerEncoded)
decoded = word_enc.inverse_transform(output)
# output management
print(decoded)
allText = primer + " " + decoded

# data into and out of generative model pt2 and output management
for i in range(textLength):
    primer = decoded
    primerEncoded = word_enc.transform(primer)
    output = textClass.predict(primerEncoded)
    decoded = word_enc.inverse_transform(output)
    print(decoded)
    allText = allText + " " + decoded

allText.replace("\\", "")

# write out of generated text 
with open("OutputTexts/"+str(desiredFileName)+".txt" , 'w') as g:
    g.write(allText)
    g.close()