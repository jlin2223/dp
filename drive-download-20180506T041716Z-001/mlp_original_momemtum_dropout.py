'''
Author: Junpeng Lin
Create Data: April 27, 2018
'''
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

data_sample = np.zeros([2000, data.shape[1]])
data_sample_label = np.zeros(2000)
i = 0
while i < 2000:

    it = np.random.randint(data.shape[0])
    data_sample[i] = data[it]
    data_sample_label[i] = data_label[it]
    #print(i)
    i += 1


class Activation(object):
    def __relu(self, x):
        return np.maximum(0.001 * x, 0.05 * x)

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
              """
              Typical hidden layer of a MLP: units are fully-connected and have
              sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
              and the bias vector b is of shape (n_out,).

              NOTE : The nonlinearity used here is tanh

              Hidden unit activation is given by: tanh(dot(input,W) + b)

              :type n_in: int
              :param n_in: dimensionality of input

              :type n_out: int
              :param n_out: number of hidden units

              :type activation: string
              :param activation: Non linearity to be applied in the hidden
                                 layer
              """
              self.input = None
              self.activation = Activation(activation).f
              self.activation_deriv = Activation(activation).f_deriv
              # end-snippet-1

              self.v_w = 0
              self.v_b = 0

              self.num_layer = num_layer
              self.cache={}
              self.cache['cache'+np.str(num_layer)] = None

              # `W` is initialized with `W_values` which is uniformely sampled
              # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
              # Note : optimal initialization of weights is dependent on the
              #        activation function used (among other things).
              #        For example, results presented in [Xavier10] suggest that you
              #        should use 4 times larger initial weights for sigmoid
              #        compared to tanh
              #        We have no info for other function, so we use the same as
              #        tanh.
              self.W = np.random.uniform(
                     low=-np.sqrt(6 / (n_in + n_out)),
                     high=np.sqrt(6 / (n_in + n_out)),
                     size=(n_in, n_out)
              )
              if activation == 'logistic':
                     W_values *= 4

              self.b = np.zeros(n_out, )

              #print(self.W.shape)
              #print('w=',self.W)

              self.grad_W = np.zeros(self.W.shape)
              self.grad_b = np.zeros(self.b.shape)

       def dropout_forward(self, x, dropout_param):
           """
           performs the forward pass for (inverted) dropout.
           Inputs:
           -x: Input data, of any shape
           -dropout_param: A dictionary with the following keys: (p,test/train,seed)
           Outputs:(out,self.cache)

           :param x:
           :param dropout_param:
           :return:
           """

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

           elif mode == 'test':
               # During prediction no mask is used
               mask = None
               out = x
           # Save mask and dropout parameters for backpropagation
           self.cache['dropout_param'] = dropout_param
           self.cache['mask'] = mask

           # Convert "out" type and return output and cache
           out = out.astype(x.dtype, copy=False)
           return out

       def forward(self, input, dropout_param):
              # dropout function
              #p  = dropout_param['p']
              out = self.dropout_forward(input, dropout_param)


              lin_output = np.dot(out, self.W) + self.b
              self.output = (
                     lin_output if self.activation is None
                     else self.activation(lin_output)
              )
              self.input = input
              return self.output, self.W

       def dropout_backward(self, dout):
           """
           Perform the backward pass for (inverted) dropout
           Inputs:
           -dout: Upstream derivatives, of any shape
           -cache:(dropout_param,mask) from dropout_forward.
           :param dout:
           :param cache:
           :return:
           """
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
              #print('self.input in backwards', self.input.shape)
              #print('delta shape in backwards', delta.shape)

              self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) + lamdb * self.W / len(self.input) #atleast_2d: transfer data into 2 dimension
              #self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
              # drouput function
              self.grad_W = self.dropout_backward(self.grad_W)
              #self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
              self.grad_b = delta

              d = delta.dot(self.W.T)
              delta_ = d * self.activation_deriv(self.input)

              return delta_


class MLP:
       """
       """

       def __init__(self, layers, activation='relu'):
              """
              :param layers: A list containing the number of units in each layer.
              Should be at least two values
              :param activation: The activation function to be used. Can be
              "logistic" or "tanh"
              """
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
              error = y_hat[y] - np.max(y_hat)
              loss = np.sum(error ** 2) /len(y_hat) + regularization_cost
              #loss = np.sum(error ** 2) / len(y_hat)
              # write down the delta in the last layer
              #delta = error * activation_deriv(np.max(y_hat))
              delta = error * activation_deriv(y_hat)

              # return loss and delta
              return loss, delta

       def backward(self, delta, lambd):
              for layer in reversed(self.layers):
                     delta = layer.backward(delta, lambd)

       def update(self,learning_rate, momemtum_rate):
              for layer in self.layers:
                     #print('grad_W', layer.grad_W)
                     layer.v_w = momemtum_rate * layer.v_w + (1-momemtum_rate) * layer.grad_W
                     layer.v_b = momemtum_rate * layer.v_b + (1-momemtum_rate) * layer.grad_b
                     layer.W -= learning_rate * layer.v_w
                     layer.b -= learning_rate * layer.v_b
                     #print('grad_b', layer.grad_b)


       def fit(self, X, y, learning_rate=0.1, epochs=100, lamdb=0.01, momemtum_rate=0.5, dropout_param={'p':0.5 ,'mode':'train','seed':6}):
              """
              Online learning.
              :param X: Input data or features
              :param y: Input targets
              :param learning_rate: parameters defining the speed of learning
              :param epochs: number of times the dataset is presented to the network for learning, 假设100次能让函数收敛
              """
              X = np.array(X)
              y = np.array(y)
              to_return = np.zeros(epochs)

              for k in range(epochs):
                     loss = np.zeros(X.shape[0]) # X.shape[0] denotes the number of data and x.shape[1] denotes the dimension of data
                     for it in range(X.shape[0]):
                            i = np.random.randint(X.shape[0]) # it will iterate all the data in an array

                            # forward pass
                            y_hat = self.forward(X[i], dropout_param)

                            # backward pass
                            loss[it], delta = self.criterion_MSE(y[i], y_hat, lamdb)
                            #print('loss[',it,']=',loss[it])
                            self.backward(delta, lamdb)

                            # update
                            self.update(learning_rate, momemtum_rate)
                            print('finish: iteration=', k+1,'  ', it/X.shape[0] * 100,'%')
                     to_return[k] = np.mean(loss)
              return to_return

       def predict(self, x):
              x = np.array(x)
              output = np.zeros(x.shape[0])
              for i in np.arange(x.shape[0]):
                     result = nn.forward(x[i, :], dropout_param={'p':0.5,'mode':'test','seed':6})
                     #print('result', result)
                     output[i] = np.argmax(result)
              return output

### Try different MLP models


nn = MLP([128,90,70,35,20,10], 'relu')     # auto initialize three class: MLP, HiddenLayer, Activation
input_data = data_sample
output_data = data_sample_label
### Try different learning rate and epochs

MSE = nn.fit(input_data, output_data, learning_rate=0.01, epochs=1, lamdb=10, momemtum_rate=0.9,dropout_param={'p':0.5 ,'mode':'train','seed':6})
print('loss:%f'%MSE[-1])

# pl.figure(figsize=(15,4))
# pl.plot(MSE)
# pl.show()

output = nn.predict(input_data)

print('predict:',output)
print('answer:',output_data)

m = output == output_data
correct = np.sum(m==True)
print('rate of correct', correct/output_data.shape)
# pl.figure(figsize=(8,6))
# pl.scatter(output_data, output, s=100)
# pl.xlabel('Targets')
# pl.ylabel('MLP output')
# pl.show()