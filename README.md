# TextGenerator
### Text generating AI agent from the University of Otago's COSC343 paper. 

Two files to interact with are "SRNTrain.py" and "loadingModel.py".
The former reads in a body of text which the model is trained on, and the later calls upon that model to predict a new body of text based off a prompt string.

These models were trained on Karl Marx's Communist Manifesto, sourced from: https://www.gutenberg.org/files/61/61.txt, and spam email I received

Key parts of the training code:
![training](https://user-images.githubusercontent.com/80669114/119070205-a0e2ff80-ba3b-11eb-9889-7c2f2943112a.png)

Key parts of the generating code:
![generation](https://user-images.githubusercontent.com/80669114/119070210-a2acc300-ba3b-11eb-87d7-3441494e72fc.png)
