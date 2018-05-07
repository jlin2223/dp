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

data_sample = np.zeros([50, data.shape[1]])
data_sample_label = np.zeros(50)
i = 0
while i < 50:

    it = np.random.randint(data.shape[0])
    data_sample[i] = data[it]
    data_sample_label[i] = data_label[it]
    #print(i)
    i += 1


class Activation(object):
    def __relu(self, x):
        return np.maximum(0.0, x)

    def __relu_deriv(self, a):
        dx = np.ones_like(a)
        dx[a <= 0] = 0
        return dx

    def __softmax(self, x):
        log_c = np.exp(- x.max())

        exp_scores = log_c * np.exp(x)

        scores_sum = np.sum(exp_scores)

        probs = exp_scores / scores_sum

        return probs

    def __softmax_deriv(self, a):
        x = np.array(a)
        return x-x**2

    def __init__(self,activation='relu'):
        if activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv


class HiddenLayer(object):
       def __init__(self, n_in, n_out, W=None, b=None,
                    activation='relu'):
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

       def forward(self, input):
              '''
              :type input: numpy.array
              :param input: a symbolic tensor of shape (n_in,)
              '''
              #print('w.shape=',self.W.shape,', input shape=',input.shape)

              lin_output = np.dot(input, self.W) + self.b
              self.output = (
                     lin_output if self.activation is None
                     else self.activation(lin_output)
              )
              self.input = input
              return self.output, self.W

       def backward(self, delta, lambd):
              self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) + lambd * self.W / len(self.input) #atleast_2d: transfer data into 2 dimension
              #self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
              self.grad_b = delta
              # return delta_ for next layer
              d = delta.dot(self.W.T)
              delta_ = d * self.activation_deriv(self.input)
              print('d.shape=',d.shape)
              print('self.shape=',self.input.shape)
              print('delta_.shape', delta_.shape)
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
              for i in range(len(layers) - 1):  # equip the layer
                     # the first parameter is the number of dimension of data and the second is the number of neurons in a layer
                     # in this case, 2 is the dimensional of data, 1 denotes neurons unit.
                     self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation='relu'))



       def forward(self, input):
              #l_W = []
              for layer in self.layers:
                     output, w = layer.forward(input)
                     input = output
                     #l_w = np.sum(np.square(w))
                     #l_W.append(l_w)
                     self.W.append(w)
              return output

       def criterion_MSE(self, y, y_hat, lambd):
              activation_deriv = Activation(self.activation).f_deriv

              # L2 regularization cost
              l_W = []
              for w in self.W:
                  l_w = np.sum(np.dot(w, w.T))
                  l_W.append(l_w)
              regularization_cost = lambd * np.sum(l_W) /len(y_hat)/2

              # MSE
              y = np.int(y)
              dW = np.zeros(10, )
              num_classes = y_hat.shape[0]
              loss = 0.0

              t = y_hat - np.max(y_hat)
              exps = np.exp(t)
              p =  exps/ np.sum(exps)
              print('sum of p = ', np.sum(p))
              log_likelihood = -np.log(p[y])
              loss = np.sum(log_likelihood) + regularization_cost

              delta = p
              delta[y] -= 1

              #loss = np.sum(error ** 2) /len(y_hat) + regularization_cost
              # write down the delta in the last layer
              #delta = error * activation_deriv(y_hat)
              # return loss and delta
              return loss, delta

       def backward(self, delta, lambd):
              for layer in reversed(self.layers):
                     delta = layer.backward(delta, lambd)

       def update(self, lr):
              for layer in self.layers:
                     layer.W += lr * layer.grad_W
                     layer.b += lr * layer.grad_b

       def fit(self, X, y, learning_rate=0.1, epochs=100, lamdb=0.01):
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
                            y_hat = self.forward(X[i])

                            # backward pass
                            loss[it], delta = self.criterion_MSE(y[i], y_hat, lamdb)
                            #print('loss[',it,']=',loss[it])
                            self.backward(delta, lamdb)

                            # update
                            self.update(learning_rate)
                            print('finish: iteration=', k + 1, '  ', it / X.shape[0] * 100, '%')
                     to_return[k] = np.mean(loss)
              return to_return

       def predict(self, x):
              x = np.array(x)
              output = np.zeros(x.shape[0])
              for i in np.arange(x.shape[0]):
                     result = nn.forward(x[i, :])
                     output[i] = np.argmax(result)
              return output

### Try different MLP models


nn = MLP([128,100,80,60,30,10], 'relu')     # auto initialize three class: MLP, HiddenLayer, Activation
input_data = data_sample
output_data = data_sample_label
### Try different learning rate and epochs
MSE = nn.fit(input_data, output_data, learning_rate=0.1, epochs=1, lamdb=0.99)
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