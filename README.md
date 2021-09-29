# TextGenerator
### Text generating AI agent from the University of Otago's COSC343 paper. 

Two files to interact with are "SRNTrain.py" and "loadingModel.py".
The former reads in a body of text which the model is trained on (modify the <selection> param to change between source options), and the latter calls upon that model to predict a new body of text based off a prompt string.

These models were trained on 
  - Karl Marx's Communist Manifesto, sourced from: https://www.gutenberg.org/files/61/61.txt
  - Spam email I receive 📬

Key parts of the training code:
![encoding](https://user-images.githubusercontent.com/80669114/135185261-0b83efc3-e898-4268-9b1e-d0f32e06b69f.png)


Key parts of the generating code:
![geneCode](https://user-images.githubusercontent.com/80669114/135185265-0665d5c3-29f7-4d1f-83d5-501675c2c836.png)

