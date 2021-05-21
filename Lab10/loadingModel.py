import pickle, gzip

from COSC343WordEnc import COSC343WordEnc

with gzip.open("srn\_training.save") as f:
    marxClass, word_enc = pickle.load(f)


primer = "fellows, my friends"
allText = ""
marxClass.reset()

primerEncoded = word_enc.transform(primer)

output = marxClass.predict(primerEncoded)

decoded = word_enc.inverse_transform(output)
print(decoded)
allText = primer + " " + decoded

for i in range(50):
    primer = decoded
    primerEncoded = word_enc.transform(primer)
    output = marxClass.predict(primerEncoded)
    decoded = word_enc.inverse_transform(output)
    print(decoded)
    allText = allText + " " + decoded

with open("marxOut5.txt", 'w') as g:
    g.write(allText)
    g.close()