# deep_examples
Example scripts for deep learning


Lasagne:

We modify the LSTM layer of Lasagne in order to export the cell activities. The original implementation only exports the hidden activities. We also modify the example from the recipes in order to perform online generation when using the trained network for generating texts. In the original implementation of the lstm_text_generation.py script, the cell and hidden states are reset to 0 and text generation works by feeding a sequence (with the seed or generated characters) over a sliding window. The modification starts with an initial setting of the cell and hidden units but then proceeds online by being fed by the seed characters and then the generated characters.
