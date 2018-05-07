import numpy as np
import h5py
import math

with h5py.File('/Users/david/PycharmProjects/Assignment1/data/train_128.h5','r') as H:
    data = np.copy(H['data'])
    #print(data.shape[0])
    #np.savetxt('/Users/david/PycharmProjects/Assignment1/data/train_128.csv', data, delimiter=',')
with h5py.File('/Users/david/PycharmProjects/Assignment1/data/train_label.h5','r') as H_l:
    data_label = np.copy(H_l['label'])
    #print(data_label.shape)

data_sample = np.zeros([1000, data.shape[1]])
data_sample_label = np.zeros(1000)
i = 0
while i < 1000:

    it = np.random.randint(data.shape[0])
    data_sample[i] = data[it]
    data_sample_label[i] = data_label[it]
    #print(i)
    i += 1




class Activation(object):

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2
    def __relu(self, x):
        return np.maximum(0.001 * x, 0.5 * x)

    def __relu_deriv(self, a):
        dx = np.ones_like(a)
        dx[a <= 0] = 0
        return dx

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        # a = logistic(x)
        return  a * (1 - a )
    def __init__(self,activation='relu'):
        if activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv


class HiddenLayer(object):
       def __init__(self, n_in, n_out, W=None, b=None,
                    activation='relu', num_layer=1):

              self.input = None
              self.activation = Activation(activation).f
              self.activation_deriv = Activation(activation).f_deriv
              # end-snippet-1

              self.v_w = 0
              self.v_b = 0

              self.num_layer = num_layer
              self.cache={}
              self.cache['cache'+np.str(num_layer)] = None


              self.W = np.random.uniform(
                     low=-np.sqrt(6 / (n_in + n_out)),
                     high=np.sqrt(6 / (n_in + n_out)),
                     size=(n_in, n_out)
              )


              self.b = np.zeros(n_out, )



              self.grad_W = np.zeros(self.W.shape)
              self.grad_b = np.zeros(self.b.shape)

       def dropout_forward(self, x, dropout_param):


           # Get the current dropout mode, p, seed
           p, mode = dropout_param['p'], dropout_param['mode']
           if 'seed' in dropout_param:
               np.random.seed(dropout_param['seed'])

           # Initialization of outputs and mask

           mask = None
           out = x

           if mode == 'train':
               # Create an apply mask (normally p=0.5 for half of neurons), we scale all
               # by p to void having to multiply by p on backpropagation, this is called inverted dropout
               mask = (np.random.rand(*x.shape) < p) / p
               out = x * mask
               #print('mask.shape', mask.shape)
           elif mode == 'test':
               # During prediction no mask is used
               mask = None
               out = x
           # Save mask and dropout parameters for backpropagation
           self.cache['dropout_param'] = dropout_param
           self.cache['mask'] = mask

           # Convert "out" type and return output and cache
           out = out.astype(x.dtype, copy=False)
           return out, mask, mode

       def forward(self, input, dropout_param):
              # dropout function
              #p  = dropout_param['p']
              out, mask, mode = self.dropout_forward(input, dropout_param)


              lin_output = np.dot(out, self.W) + self.b
              self.output = (
                     lin_output if self.activation is None
                     else self.activation(lin_output)
              )
              self.input = input
              #print('self.W .shape',self.W.shape)

              if mode == 'train':
                return self.output, (mask * self.W.T).T
              elif mode == 'test':
                return self.output, self.W

       def dropout_backward(self, dout):

           # Recover dropout parameters(p, mask,mode) from cache
           dropout_param, mask = self.cache['dropout_param'], self.cache['mask']
           mode = dropout_param['mode']

           dx = None
           # Back propagate (Dropout layer has no parameters just input X)
           if mode == 'train':
               # Just back propagate dout from the neurons that were used during dropout
               #print('dout.shape', dout.shape)
               #print('mask shape', np.atleast_2d(mask).shape)
               #print('mask.T shape', np.atleast_2d(mask).T.shape)

               dx = np.atleast_2d(mask).T * dout
               #print('dx.shape', dx.shape)
           elif mode == 'test':
               dx = dout

           # Return dx
           return dx

       def backward(self, delta, lamdb):
              dropout_param, mask = self.cache['dropout_param'], self.cache['mask']
              # print('self.input in backwards', self.input.shape)
              # print('delta shape in backwards', np.atleast_2d(delta).shape)
              # print('mask,shape', mask.shape)
              # print('self.W', self.W.shape)
              mode = dropout_param['mode']
              if mode == 'train':
                    self.grad_W = np.atleast_2d(mask * self.input).T.dot(np.atleast_2d(delta)) + lamdb * (mask * self.W.T).T / len(self.input) #atleast_2d: transfer data into 2 dimension
              elif mode == 'test':
                  self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) + lamdb * self.W / len(
                      self.input)  # atleast_2d: transfer data into 2 dimension
              #self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
              # drouput function
              #self.grad_W = self.dropout_backward(self.grad_W)
              #self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
              self.grad_b = delta

              d = delta.dot(self.W.T)
              delta_ = d * self.activation_deriv(self.input)

              return delta_


class MLP:


       def __init__(self, layers, activation='relu'):

              ### initialize layers
              self.layers = []

              self.params = []
              self.W = []
              self.activation = activation



              print('number of layer:', len(layers))
              for i in range(len(layers)-1):  # equip the layer
                     # the first parameter is the number of dimension of data and the second is the number of neurons in a layer
                     # in this case, 2 is the dimensional of data, 1 denotes neurons unit.
                     self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation='relu', num_layer=i))
                     #self.layers.append(HiddenLayer(layers[i+1], layers[i + 2], activation='relu', num_layer=i))
                     #self.layers.append(HiddenLayer(layers[i + 2], layers[i + 3], activation='tanh', num_layer=i))
                     self.neuron = layers[-1]

       def forward(self, input, dropout_param):
              #l_W = []
              for layer in self.layers:
                     output, w = layer.forward(input, dropout_param)
                     input = output
                     #l_w = np.sum(np.square(w))
                     #l_W.append(l_w)
                     self.W.append(w)
              return output

       def softmax_predict(self, X):
           exp_scores = np.exp(X-np.max(X))
           #print('exp_scores.shape = ', exp_scores.shape)
           return exp_scores / np.sum(exp_scores)

       def loss(self, X, y):
           num_examples = X.shape[0]
           #print('y_hat.shape = ', X.shape)
           probs = self.softmax_predict(X)

           correct_logprobs = -np.log(probs[y])
           #data_loss = np.sum(corect_logprobs)
           return correct_logprobs

       def diff(self, X, y):
           num_examples = X.shape[0]
           probs = self.softmax_predict(X)
           probs[y] -= 1
           return probs

       def criterion_MSE(self, y, y_hat, lambd):
              activation_deriv = Activation(self.activation).f_deriv
              y = np.int(y)

              # L2 regularization cost
              l_W = []
              for w in self.W:
                  l_w = np.sum(np.dot(w, w.T))
                  l_W.append(l_w)
              regularization_cost = lambd * np.sum(l_W) /len(y_hat)/2

              # MSE
              loss = self.loss(y_hat, y) + regularization_cost

              delta = self.diff(y_hat, y)
              #error = y_hat[y] - np.max(y_hat)
              #loss = np.sum(error ** 2) /len(y_hat) + regularization_cost
              #loss = np.sum(error ** 2) / len(y_hat)
              # write down the delta in the last layer
              #delta = error * activation_deriv(np.max(y_hat))
              #delta = error * activation_deriv(y_hat)

              # return loss and delta
              return loss, delta

       def backward(self, delta, lambd):
              for layer in reversed(self.layers):
                     delta = layer.backward(delta, lambd)

       def update_mini(self,learning_rate, momemtum_rate):
              for layer in self.layers:
                     #print('grad_W', layer.grad_W)
                     layer.v_w = momemtum_rate * layer.v_w + (1-momemtum_rate) * layer.grad_W
                     layer.v_b = momemtum_rate * layer.v_b + (1-momemtum_rate) * layer.grad_b
                     layer.W -= learning_rate * layer.v_w
                     layer.b -= learning_rate * layer.v_b
                     #print('grad_b', layer.grad_b)

       def random_mini_batches(self, X, y, mini_batch_size=16, seed_m=0):

           np.random.seed(seed_m)  # To make your "random" minibatches the same as ours
           m = X.shape[0]  # number of training examples
           mini_batches = []

           # Step 1: Shuffle (X, Y)
           permutation = list(np.random.permutation(m))
           shuffled_X = X[permutation]
           shuffled_y = y[permutation].reshape(1, m)

           # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
           num_complete_minibatches = math.floor(
               m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
           for k in range(0, num_complete_minibatches):
               ### START CODE HERE ### (approx. 2 lines)
               mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size]
               mini_batch_y = shuffled_y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
               ### END CODE HERE ###
               mini_batch = (mini_batch_X, mini_batch_y)
               mini_batches.append(mini_batch)

           # Handling the end case (last mini-batch < mini_batch_size)
           if m % mini_batch_size != 0:
               ### START CODE HERE ### (approx. 2 lines)
               mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:]
               mini_batch_y = shuffled_y[:, num_complete_minibatches * mini_batch_size:]
               ### END CODE HERE ###
               mini_batch = (mini_batch_X, mini_batch_y)
               mini_batches.append(mini_batch)

           return mini_batches



       def fit(self, X, y, learning_rate=0.1, epochs=100, lamdb=0.01, momemtum_rate=0.5, dropout_param={'p':0.5 ,'mode':'train','seed':6},mini_batch_size=64):

            X = np.array(X)
            y = np.array(y)
            to_return = 0
            seed_m=10
            average_loss = 0
            for k in range(epochs):
                    seed_m = seed_m + 1
                    minibatches = self.random_mini_batches(X, y, mini_batch_size, seed_m)


                    mini_count = 1
                    for minibatch in minibatches:


                        # Select a minibatch
                        (minibatch_X, minibatch_y) = minibatch

                        loss = np.zeros(minibatch_X.shape[
                                            0])  # X.shape[0] denotes the number of data and x.shape[1] denotes the dimension of data
                        #print('minibatch_x', minibatch_X)
                        #print('minibatch_y', minibatch_y)
                        for it in range(minibatch_X.shape[0]):
                                i = np.random.randint(minibatch_X.shape[0]) # it will iterate all the data in an array

                            # forward pass
                                y_hat = self.forward(minibatch_X[i], dropout_param)

                            # backward pass
                                loss[it], delta = self.criterion_MSE(minibatch_y[:,i], y_hat, lamdb)
                                self.backward(delta, lamdb)
                                self.update(learning_rate)
                                print('finish epoch:',k,'batch:', mini_count,'data:',it)

                        average_loss += loss
                            #print('loss[',it,']=',loss[it])
                        mini_count += 1
                    # update


                        self.update_mini(learning_rate, momemtum_rate)

            to_return = (average_loss) / X.shape[0]

            return to_return
       def update(self, lr):
              for layer in self.layers:
                     layer.grad_W += lr * layer.grad_W
                     layer.grad_b += lr * layer.grad_b
       def predict(self, x):
              x = np.array(x)
              output = np.zeros(x.shape[0])
              for i in np.arange(x.shape[0]):
                     result = nn.forward(x[i, :], dropout_param={'p':0.5,'mode':'test','seed':6})
                     #print('result', result)
                     output[i] = np.argmax(result)
              return output

### Try different MLP models


nn = MLP([128,20,10], 'relu')     # auto initialize three class: MLP, HiddenLayer, Activation
input_data = data_sample
output_data = data_sample_label
### Try different learning rate and epochs

MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=1, lamdb=0.5, momemtum_rate=0.9,dropout_param={'p':0.3 ,'mode':'train','seed':5},mini_batch_size=100)




output = nn.predict(input_data)

print('predict:',output)
print('answer:',output_data)
print('loss:%f'%MSE[-1])
m = output == output_data
correct = np.sum(m==True)
print('accuracy: %.2f%%'%((correct/output_data.shape)*100))
#print('rate of correct', correct/output_data.shape)
