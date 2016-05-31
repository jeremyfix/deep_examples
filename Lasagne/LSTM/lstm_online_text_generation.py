# coding: utf-8
'''
Recurrent network example.  Trains a 2 layered LSTM network to learn
text from a user-provided input file. The network can then be used to generate
text using a short string as seed (refer to the variable generation_phrase).
This example is partly based on Andrej Karpathy's blog
(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and a similar example in the Keras package (keras.io).
The inputs to the network are batches of sequences of characters and the corresponding
targets are the characters in the text shifted to the right by one. 
Assuming a sequence length of 5, a training point for a text file
"The quick brown fox jumps over the lazy dog" would be
INPUT : 'T','h','e',' ','q'
OUTPUT: 'u'

The loss function compares (via categorical crossentropy) the prediction
with the output/target.

Also included is a function to generate text using the RNN given the first 
character.  

About 20 or so epochs are necessary to generate text that "makes sense".

Written by @keskarnitish
Pre-processing of text uses snippets of Karpathy's code (BSD License)

Modified by @jeremyfix
The text generation part is now online rather than batch based.

'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import urllib2 #For downloading the sample text file. You won't need this if you are providing your own file.
import lstm

try:
    in_text = urllib2.urlopen('https://s3.amazonaws.com/text-datasets/nietzsche.txt').read()
    #You can also use your own file
    #The file must be a simple text file.
    #Simply edit the file name below and uncomment the line.  
    #in_text = open('your_file.txt', 'r').read()
    in_text = in_text.decode("utf-8-sig").encode("utf-8")
except Exception as e:
    print("Please verify the location of the input file/URL.")
    print("A sample txt file can be downloaded from https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    raise IOError('Unable to Read Text')

generation_phrase = "The quick brown fox jumps" #This phrase will be used as seed to generate text.

#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 
chars = list(set(in_text))
data_size, vocab_size = len(in_text), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 50

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 128

# Optimization learning rate
LEARNING_RATE = .002

# Decay of the learning rate
LEARNING_RATE_DECAY_RATE = 0.97
LEARNING_RATE_DECAY_AFTER = 10

# Decay rate of rmsprop
DECAY_RATE = 0.95

# All gradients above this will be clipped
GRAD_CLIP = 5

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 50


def gen_data(p, batch_size = BATCH_SIZE, data=in_text, return_target=True):
    '''
    This function produces a semi-redundant batch of training samples from the location 'p' in the provided string (data).
    For instance, assuming SEQ_LENGTH = 5 and p=0, the function would create batches of 
    5 characters of the string (starting from the 0th character and stepping by 1 for each semi-redundant batch)
    as the input and the next character as the target.
    To make this clear, let us look at a concrete example. Assume that SEQ_LENGTH = 5, p = 0 and BATCH_SIZE = 2
    If the input string was "The quick brown fox jumps over the lazy dog.",
    For the first data point,
    x (the inputs to the neural network) would correspond to the encoding of 'T','h','e',' ','q'
    y (the targets of the neural network) would be the encoding of 'u'
    For the second point,
    x (the inputs to the neural network) would correspond to the encoding of 'h','e',' ','q', 'u'
    y (the targets of the neural network) would be the encoding of 'i'
    The data points are then stacked (into a three-dimensional tensor of size (batch_size,SEQ_LENGTH,vocab_size))
    and returned. 
    Notice that there is overlap of characters between the batches (hence the name, semi-redundant batch).
    '''
    x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n,i,char_to_ix[data[p+ptr+i]]] = 1.
        if(return_target):
            y[n] = char_to_ix[data[p+ptr+SEQ_LENGTH]]
    return x, np.array(y,dtype='int32')



def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size), name="input")

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    cell_init_forward_1 = lasagne.layers.InputLayer(shape=(None, N_HIDDEN), name="cell_init_forward_1")
    hid_init_forward_1 = lasagne.layers.InputLayer(shape=(None, N_HIDDEN), name="hid_init_forward_1")

    l_forward_1 = lstm.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh, cell_init = cell_init_forward_1, hid_init = hid_init_forward_1, name="lstm1")

    # The layer is sliced to keep only the hidden activities and discarding the 
    # the cell activities
    l_forward_1_slice = lasagne.layers.SliceLayer(l_forward_1,1,0,name="lstm1_slice")

    cell_init_forward_2 = lasagne.layers.InputLayer(shape=(None, N_HIDDEN), name="cell_init_forward_2")
    hid_init_forward_2 = lasagne.layers.InputLayer(shape=(None, N_HIDDEN), name="hid_init_forward_2")

    l_forward_2 = lstm.LSTMLayer(
        l_forward_1_slice, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh, cell_init = cell_init_forward_2, hid_init = hid_init_forward_2, name="lstm2")

    # The l_forward layer creates an output of dimension (2, batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction of the hidden units, we isolate that quantity and feed it to the next layer. 
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    l_forward_2_slice = lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_forward_2,1,0), -1, 1, name="lstm2_slice")

    # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(l_forward_2_slice, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax, name="output")

    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    lr = theano.shared(np.cast['float32'](LEARNING_RATE))
    updates = lasagne.updates.rmsprop(cost, all_params, learning_rate=lr, rho=DECAY_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, cell_init_forward_1.input_var, hid_init_forward_1.input_var, cell_init_forward_2.input_var, hid_init_forward_2.input_var], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values, cell_init_forward_1.input_var, hid_init_forward_1.input_var, cell_init_forward_2.input_var, hid_init_forward_2.input_var], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    # probs = theano.function([l_in.input_var, cell_init_forward_1.input_var, hid_init_forward_1.input_var, cell_init_forward_2.input_var, hid_init_forward_2.input_var],network_output,allow_input_downcast=True)
    probs_with_state= theano.function([l_in.input_var, cell_init_forward_1.input_var, hid_init_forward_1.input_var, cell_init_forward_2.input_var, hid_init_forward_2.input_var], [network_output, lasagne.layers.get_output(l_forward_1), lasagne.layers.get_output(l_forward_2)],allow_input_downcast=True)


    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict. 

    def try_it_out(N=200):
        '''
        This function uses the user-provided string "generation_phrase" and current state of the RNN generate text.
        It generates the text character by character in an online fashion. 
       '''

	assert(len(generation_phrase) >= 1) # We need at least one character for as input

        sample_ix = [] 

	cell1 = np.zeros((1, N_HIDDEN))
	hid1 = np.zeros((1, N_HIDDEN))
	cell2 = np.zeros((1, N_HIDDEN))
	hid2 = np.zeros((1, N_HIDDEN))
	proba = np.zeros((vocab_size,))

        # Bring in the seed
	for c in generation_phrase:
		ix = char_to_ix[c]
		x = np.zeros((1, 1, vocab_size))
		x[0, 0, ix] = 1
                proba, state1, state2 = probs_with_state(x, cell1, hid1, cell2, hid2)
		cell1[:], hid1[:] = state1[0], state1[1]
		cell2[:], hid2[:] = state2[0], state2[1]
		

 	# And then we generate the following
	for i in range(N):
		ix = np.argmax(proba)
		x = np.zeros((1, 1, vocab_size))
		x[0, 0, ix] = 1
                proba, state1, state2 = probs_with_state(x, cell1, hid1, cell2, hid2)
		cell1[:], hid1[:] = state1[0], state1[1]
		cell2[:], hid2[:] = state2[0], state2[1]
		sample_ix.append(np.argmax(proba))

        random_snippet = generation_phrase + ''.join(ix_to_char[ix] for ix in sample_ix)    
        print("----\n %s \n----" % random_snippet)


    
    print("Training ...")
    print("Seed used for text generation is: " + generation_phrase)
    cell_init = np.zeros((BATCH_SIZE, N_HIDDEN))
    hid_init = np.zeros((BATCH_SIZE, N_HIDDEN))
    p = 0
    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):
            try_it_out() # Generate text online 
        
	    if((it/data_size * BATCH_SIZE) > LEARNING_RATE_DECAY_AFTER):
		lr = lr * LEARNING_RATE_DECAY_RATE
 		print("Decayed learning rate by a factor of %f to %f " % (LEARNING_RATE_DECAY_RATE, lr))
    
            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)
                
                #print(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1 
                if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    p = 0;
                

                avg_cost += train(x, y, cell_init, hid_init, cell_init, hid_init)
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
                    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
