import h5py
import numpy as np
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


 ### ReLU activation ########
class Activation(object):
    def __relu(self, x):
        #return np.maximum(-0.00000000000001*x, x)
        return np.maximum(0,x)

    def __relu_deriv(self, a):
        dx = np.array(a)
        dx[dx < 0] = 0.0
        dx[dx> 0] = 1.0
        return dx



    def __init__(self,activation='relu'):
        if  activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        else:
            if activation == 'tanh':
                self.f = self.__tanh
                self.f_deriv = self.__tanh_deriv




class HiddenLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None,
                 activation='relu'):

        self.input = None
        self.activation = Activation(activation).f
        self.activation_deriv = Activation(activation).f_deriv

        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        #print(self.W)

        self.b = np.zeros(n_out, )

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input):

        lin_output = np.dot(input,self.W) + self.b
        #print(self.W)
        #print(input)
        #print(lin_output)
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        #print(self.output)
        return self.output, self.W

    def backward(self, delta, lamdb):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) + lamdb * self.W / len(self.input)
        self.grad_b = delta
        # return delta_ for next layer
        delta_ = delta.dot(self.W.T) * self.activation_deriv(self.input)

        return delta_





class MLP:

    def __init__(self, layers, activation='relu'):

        self.layers = []
        self.params = []
        self.neuron = 0
        self.W = []
        #self.neuron2 = 0

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation= 'relu'))
            self.neuron = layers[i+1]
            print('relu')

        #self.layers.append(HiddenLayer(layers[len(layers)-2],layers[len(layers)-1], activation = 'softmax'))
        #self.neuron = layers[len(layers)-1]
        #print('softmax')

    def forward(self, input):
        for layer in self.layers:
            output, w = layer.forward(input)
            input = output
            self.W.append(w)
            #print('forward:', input)
        return output



    def criterion_MSE(self, y, y_hat, lamdb):
        activation_deriv = Activation(self.activation).f_deriv

        ' L2 regularization cost'
        l_W =[]
        for w in self.W:
            l_w = np.sum(np.dot(w, w.T))
            l_W.append(l_w)
        regularization_cost = lamdb * np.sum(l_W) / len(y_hat) / 2

        #new_y_hat = np.max(y_hat)
        #new_y = y_hat[np.int(y)]
        error = y_hat[np.int(y)] - np.max(y_hat)
        #error = new_y - new_y_hat
        loss = error ** 2 + regularization_cost

        # y_hat2=y_hat.reshape( ,1)
        delta = error * activation_deriv(y_hat)
        #print('error:', error)
        #print('y_hat:', y_hat)
        #print('delta: ', delta)

        return loss, delta



    def backward(self, delta, lamdb):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lamdb)

    def update(self, lr):
        for layer in self.layers:
            layer.W += lr * layer.grad_W
            layer.b += lr * layer.grad_b


    def random_mini_batches(self, X, y, mini_batch_size=16, seed=0):

        np.random.seed(seed)  # To make your "random" minibatches the same as ours
        m = X.shape[0]  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation]
        shuffled_y = y[permutation].reshape(1,m)

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
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





    def fit(self, X, y, learning_rate=0.1, epochs=100,mini_batch_size=64, lamdb =0.9):

        X = np.array(X)
        y = np.array(y)
        seed = 10
        to_return = np.zeros(epochs)


        for k in range(epochs):
            seed = seed + 1
            minibatches = self.random_mini_batches(X, y, mini_batch_size, seed)

            loss = np.zeros([X.shape[0],self.neuron])
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_y) = minibatch
                #print('minibatch_x', minibatch_X)
                #print('minibatch_y', minibatch_y[:,4])

                #loss = np.zeros([minibatch_X.shape[0],self.neuron])

                for it in range(minibatch_X.shape[0]):
                    i = np.random.randint(minibatch_X.shape[0])

                    # Forward pass
                    y_hat = self.forward(minibatch_X[i])

                    # backward pass
                    #print('minibatch_y:   ', minibatch_y[:,i])
                    #print('y_hat:   ', y_hat)
                    loss[it], delta = self.criterion_MSE(minibatch_y[:,i], y_hat, lamdb)
                    self.backward(delta, lamdb)

            # update
            self.update(learning_rate)

        to_return[k] = np.mean(loss)
        return to_return



    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        #print(self.layers)
        #for layer in self.layers:
            #print(layer.)
        for m in np.arange(x.shape[0]):
                y_hat = nn.forward(x[m, :])
                #print('y_hat:', y_hat)

                output[m] = np.argmax(y_hat)

        return output



### Try different MLP models
nn = MLP([128,100,80,50,20,10], 'relu')
input_data = data_sample
output_data = data_sample_label
#input_data = data
#output_data = label



### Try different learning rate and epochs
MSE = nn.fit(input_data, output_data, learning_rate=0.0001, epochs=1,mini_batch_size=100, lamdb=0.9 )
print('loss:%f'%MSE[-1])

predict_data = nn.predict(input_data)
#predict_data_int = predict_data.astype(int)
print('predict', predict_data)
#print('answer', output_data)
m = predict_data == output_data
correct = np.sum(m==True)
print('accuracy: %.2f%%'%((correct/output_data.shape)*100))




