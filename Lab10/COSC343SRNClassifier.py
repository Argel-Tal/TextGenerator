import numpy as np
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError

__author__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"

class COSC343SRNClassifier:

    # Constructor
    #
    # Inputs: hidden_layer_size - number of neurons in the single hidden layer
    #         activation - activation function to use on the hidden layer ('identity','logistic','tanh', or 'relu')
    #         solver - 'sg' or 'adam'
    #         batch_size - size of minibatches for optimizers
    #         learning_rate_init - learning parameter value for later training
    #         max_iter - maximimum number of iterations to train for
    #         verbose - if True prints info during training
    #         momentum - momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.
    #         beta_1 - decay rate for estimates of first moment vector in adam
    #         beta_2 - decay rate for estimates of second moment vector in adam
    #         epsilon - value for numerical stability in adam
    def __init__(self, hidden_layer_size=100, activation='tanh', solver='adam', batch_size='auto',
                 learning_rate_init=0.001, max_iter = 200, verbose=False,
                 momentum=0.9, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8):
        self.lr = learning_rate_init
        self.hidden_layer_size = int(hidden_layer_size)
        self.activation = activation
        self.max_iter = max_iter
        self.batch_size = batch_size
        # Initialise the hidden state to zeros
        self.context = np.zeros(self.hidden_layer_size)
        # Don't know the size of the input and output yet
        self.n_inputs_ = None
        self.n_outputs_ = None
        self.out_activation_ = 'softmax'

        if solver != 'adam' and solver != 'sgd':
            raise RuntimeError('Invalid solver \'%s\': valid options are \'s\',\'adam\'')

        self.solver = solver
        self.verbose = verbose

        self.random_state = None

        # Initialise parametes for the sgd optimiser
        self.momentum = momentum

        # Initialise parametes for the adam optimiser
        self._adam_t = 0
        self._adam_B1 = beta_1
        self._adam_B2 = beta_2
        self._adam_eps = epsilon


    # Resets the model's context
    #
    # Inputs: zeroState - boolean controlling whether context is reset to zero
    #                     or to a random vector
    def reset(self, zeroState=True):
        if zeroState:
            self.context = np.zeros(self.hidden_layer_size)
        else:
            self.context = np.random.uniform(-1,1,self.hidden_layer_size)

    # Update the model using vanilla steepest gradient optimiser
    #
    # Inputs: g - gradient vector (must be same dimension as self.params)
    def _update_sgd(self, g):
        self.g_params_ *= 0.9
        self.g_params_ += g
        self.params -= self.lr*self.g_params_

    # Update the model using adam optimiser
    #
    # Inputs: g - gradient vector (must be same dimension as self.params)
    def _update_adam(self, g):

        self._adam_t += 1
        self._adam_m *= self._adam_B1
        self._adam_m += (1.0-self._adam_B1)*g

        self._adam_v *= self._adam_B2
        self._adam_v += (1.0-self._adam_B2)*g**2

        lr = self.lr * np.sqrt(1-self._adam_B2**self._adam_t)/(1.0-self._adam_B1**self._adam_t)

        self.params -= lr*self._adam_m/(np.sqrt(self._adam_v)+self._adam_eps)

    # Slice the params vector into weight matrices and bias vectors
    #
    # Inputs: params - the parameter vector
    def _slice_params(self, params):
        i = 0
        # Weight matrix connecting inputs to the hidden layer
        Wih = np.reshape(params[i:i + self.hidden_layer_size * self.n_inputs_],
                              (self.hidden_layer_size, self.n_inputs_))
        i += self.hidden_layer_size * self.n_inputs_
        # Bias vector for the hidden layer
        bh = params[i:i + self.hidden_layer_size]
        i += self.hidden_layer_size
        # Weight matrix connecting hidden layer (from the previous input, also referred to as the context) to the
        # hidden layer of the current input
        Whh = np.reshape(params[i:i + self.hidden_layer_size ** 2],
                              (self.hidden_layer_size, self.hidden_layer_size))
        i += self.hidden_layer_size ** 2
        # Weight matrix connecting the hidden layer to the outputs
        Who = np.reshape(params[i:i + self.n_outputs_ * self.hidden_layer_size],
                              (self.n_outputs_, self.hidden_layer_size))
        i += self.n_outputs_ * self.hidden_layer_size
        # Bias vector on the outputs
        bo = params[i:i + self.n_outputs_]
        # Return the parpameters split into layred weight matrices and bias vectors
        return (Wih, bh, Whh, Who, bo)

    # Compute the output of the model
    #
    # Input: x - an NxM input matrix of N samples of M-dimensions each
    #
    # Returns: (y_out, y_hidden) - a tuple consisting of an NxK and NxU matrices, where K is the number of model's
    #                              outputs and U is the number of hidden units
    def _compute_forward(self, x):
        # Number of samples
        N = x.shape[0]

        # Slice the model parameters into layer weight matrices and bias vectors
        Wih, bh, Whh, Who, bo = self._slice_params(self.params)

        # Allocate memory for the hidden layer's activity
        #v_hidden = np.zeros((N, self.hidden_layer_size))
        # Allocate memory for the hidden layer's output
        y_hidden = np.zeros((N, self.hidden_layer_size))

        # Compute output one sample at a time (since we need context from the previous sample to compute the
        # output for the next sample)
        for n in range(N):
            # Compute the output of the hidden layer based on the input sample and the context
            y_hidden[n,:] = np.dot(x[n,:], Wih.transpose())
            y_hidden[n,:] += np.dot(self.context, Whh.transpose())
            y_hidden[n,:] += bh

            # Apply the activation function
            if self.activation == 'relu':
                y_hidden[n,y_hidden[n,:] < 0] = 0
            elif self.activation == 'logistic' or self.activation == 'sigmoid':
                y_hidden[n,:] = 1/(1+np.exp(-y_hidden[n,:]))
            elif self.activation == 'tanh':
                y_hidden[n,:] = np.tanh(y_hidden[n,:])
            elif self.activation != 'identity':
                raise ValueError(
                    "The activation '%s' is not supported. Supported activations are ('identity', 'logistic', 'tanh', 'relu')." % (
                        self.activation))

            # Update the context for the next sample
            self.context = y_hidden[n,:]

        # Compute the activity on
        y_out = np.matmul(y_hidden, Who.transpose())+bo
        y_out = y_out-np.expand_dims(np.max(y_out,axis=1),axis=1)
        y_out = np.exp(y_out)
        y_out = y_out/np.expand_dims(np.sum(y_out, axis=1), axis=1)

        return (y_out, y_hidden)

    # Trains the model
    #
    # Input: X - an NxM input matrix, where N is the number of samples and M is the dimension of each sample
    #        y - an NxK target output matrix, where N is the number of samples and K is the size of the labels
    def fit(self, X, y):

        # If the input is an M-dimensional vector, treat it as a 1xM matrix
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        # If the target output is a K-dimensional vector, treat it as a 1xK matrix
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=0)


        # If this function is being run for the first time, infer the size of the input and output and
        # set the model parameters accordingly
        if self.n_inputs_ is None:
            self.n_inputs_ = X.shape[1]
            self.n_outputs_ = y.shape[1]

            # Compute the number of parameters needed for the model
            nParams = self.hidden_layer_size*(self.n_inputs_+1)+self.hidden_layer_size*(self.hidden_layer_size+1)+self.n_outputs_*(self.hidden_layer_size+1)
            # Create the parameters vector and initialise all the components to a small random value
            self.params = np.random.randn(nParams)*0.1

        # Split the parameters into layered weight matrices and bias vectors
        Wih, bh, Whh, Who, bo = self._slice_params(self.params)

        # Softmax epsilon assures that no softmax output is exactly 0 or 1 (which would break the cost computation with
        # the logs
        _softmax_epsilon = 1e-8

        # Number of points in the training set
        N = X.shape[0]

        # Allocate memory for parameter gradient vector
        g_params = np.zeros(len(self.params))

        # Split the gradient parameters into layered weight matrices and bias vectors
        g_Wih, g_bh, g_Whh, g_Who, g_bo = self._slice_params(g_params)

        if self.batch_size=='auto':
            batch_size = np.min([200,N])
        else:
            batch_size = self.batch_size

        if self.solver == 'sgd':
            self.g_params_ = np.zeros(len(self.params))
        else:
            self._adam_m = np.zeros(len(self.params))
            self._adam_v = np.zeros(len(self.params))


        # Iterate over at most max_iter epochs
        for i in range(self.max_iter):

            # Reset the total cost
            J = 0
            # Reset the total error
            E = 0

            batch_index_start = np.random.randint(0,batch_size)


            while(batch_index_start < N):
                # Reset the model's context
                self.reset()

                batch_index_end = batch_index_start + batch_size
                if batch_index_end > N:
                    batch_index_end = N

                x_batch = X[batch_index_start:batch_index_end]
                y_batch = y[batch_index_start:batch_index_end]

                # Previous blame from the hidden layer
                e_context = np.zeros(self.hidden_layer_size)


                # Clear the delta parameters to 0
                g_params *= 0.0

                # Compute the output of all the layers of the network
                (y_out, y_hidden) = self._compute_forward(x_batch)

                # Make sure that no softmax output is exactly 0 or exactly 1
                y_out[y_out < _softmax_epsilon] = _softmax_epsilon
                y_out[y_out > 1.-_softmax_epsilon] = 1.-_softmax_epsilon

                # Compute the cross-entropy cost - should be ok for log(y_out) and log(1-yout) since y_out cannot be
                # 0 nor 1
                J += np.sum(np.sum(-y_batch *np.log(y_out),axis=1))#-(1.-y[batch_index_start:batch_index_end])*np.log(1.-y_target),axis=1))
                # Compute the number of errors
                errors =  np.abs(np.argmax(y_out,axis=1)-np.argmax(y_batch,axis=1)) != 0
                E += np.sum(errors.astype('int'))

                # Backpropagation trought time of the error, starting at the last sample
                for n in reversed(range(batch_index_end-batch_index_start)):
                    # Get the output of the hidden layer
                    yh = y_hidden[n,:]

                    # Compute the error blame from the cross-entropy cost
                    e = np.copy(y_out[n,:])
                    e[y_batch[n,:]==1.0] -= 1.0

                    # Divide the error by the number of sample in the batch
                    e = e/(batch_index_end-batch_index_start)

                    # Compute the gradient to the output weight matrix and bias
                    g_Who += np.outer(e,yh)
                    g_bo += e

                    # Compute the blame on the hidden layer output - it's the blame for this output when used as context
                    # plus the blame from the output
                    e = e_context+np.dot(e, Who)

                    # Multiply the blame by the derivative of the activation function
                    if self.activation == 'relu':
                        e[yh==0] = 0
                    elif self.activation == 'logistic' or self.activation == 'sigmoid':
                        e *= yh*(1-yh)
                    elif self.activation == 'tanh':
                        e *= (1-yh**2)
                    elif self.activation != 'identity':
                        raise ValueError("The activation '%s' is not supported. Supported activations are ('identity', 'logistic', 'tanh', 'relu')." % (self.activation))

                    # Get the context value (it's the previous output of the hidden layer) for sample n>0...and
                    # it's a zero vector for the first training sample
                    if n>0:
                        y_context = y_hidden[n-1,:]
                    else:
                        y_context = np.zeros(self.hidden_layer_size)

                    # Compute the gradient on the input to hidden layer weight matrix, the hidden layer bias vector
                    # and the context to hidden (or hidden to hidden) weight matrix
                    g_Wih += np.outer(e,x_batch[n,:])
                    g_bh += e
                    g_Whh += np.outer(e,y_context)

                    # Compute the blame on the context for the preceeding sample
                    e_context = np.dot(e, Whh)

                # Update the model with the sg or adam optimiser
                if self.solver == 'sgd':
                    self._update_sgd(g_params)
                else:
                    self._update_adam(g_params)

                batch_index_start = batch_index_end

            # Compute the accuracy of the model
            E = 1-float(E)/float(N)
            J /= float(N)

            self.loss_ = J

            # If doing perfectly, stop training
            if E==1.0:
                print("Training reached perfect score. Stopping")
                break

            # If the first, last, or one of the update epoch, show the progress
            if self.verbose:
                print("Iteration %d, loss = %.6f, score=%.6f " % (i+1,self.loss_,E))

    # Check if the model has been fitted
    def check_is_fitted(self, msg=None):
        if self.n_inputs_ is None:
            if msg is None:
                msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                       "appropriate arguments before using this method.")
            raise NotFittedError(msg % {'name': type(self).__name__})

    # Compute the output of the model
    #
    # Input: x - an NxM matrix of N input of M-dimensons each, or a M-dimensional vector representing a signle input
    #        categorical - if set to True, will otuput a zero-one vector, if set to fals, will otuput the softmax
    #                      probability distribution over the outputs
    #
    # Returns: y - an NxK matrix of outputs, or a K-dimensional vector represeting a single output
    def predict(self,X,sampled=1):

        self.check_is_fitted()

        if isinstance(X,list):
            X = np.stack(X)

        # Check if input is just a vector...
        _vec_input = False
        if len(X.shape)==1:
            #...if so, convert it to a 1xM matrix
            X = np.expand_dims(X,axis=0)
            _vec_input = True

        # Compute the softmax output of the model
        y = self._compute_forward(X)[0]

        # If categorical, for each sample pick the output with highest probability and mark it as 1, leaving rest
        # set to zero
        if sampled<=1:
            y = np.argmax(y,axis=1)
        else:
            y = np.argsort(y,axis=1)
            y = y[:,-sampled:]
            y = y[np.arange(len(y)),np.random.randint(0, sampled, len(y))]

        y = np.expand_dims(y, axis=1)
        enc = preprocessing.OneHotEncoder(categories=[range(self.n_outputs_)])
        enc.fit(y)
        y = enc.transform(y).toarray()


        # If input was in vector format, convet the matrix otuput to a vector
        if _vec_input:
            y = np.squeeze(y,axis=0)

        return y













