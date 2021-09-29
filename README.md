# TextGenerator
### Text generating AI agent from the University of Otago's COSC343 paper. 

Two files to interact with are "SRNTrain.py" and "loadingModel.py".
The former reads in a body of text which the model is trained on (modify the <selection> param to change between source options), and the latter calls upon that model to predict a new body of text based off a prompt string.

These models were trained on 
  - Karl Marx's Communist Manifesto, sourced from: https://www.gutenberg.org/files/61/61.txt
  - Spam email I receive 📬

### Key parts of the training code:
![encoding](https://user-images.githubusercontent.com/80669114/135188145-9e3bab11-c2c9-4287-8f13-4532b9ca33d0.png)

### Key parts of the generating code:
![geneCode](https://user-images.githubusercontent.com/80669114/135188164-30ce28a2-f4cf-488f-9970-e097fb17152b.png)

