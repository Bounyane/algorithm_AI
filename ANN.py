import numpy as np
class ANN:

    __activation_f=list()
    __hidden_dim=list()
    __parametres=None;

    def input(self,dim):
        self.hidden_dim.append(dim)

    def layer(self, dim, activation):
        self.hidden_dim.append(dim)
        self.activation_f.append(activation)

    def output(self ,dim):
        self.hidden_dim.append(dim)
        self.activation_f.append("softmax")

    # determinate number of hidden layer#
    def __ini_parametres(self,dim, labs):
        parametres = {}
        for i in range(len(dim) - 1):
            parametres["W" + str(i + 1)] = np.random.rand(dim[i + 1], dim[i]) * labs
            parametres["B" + str(i + 1)] = np.random.rand(dim[i + 1], 1) * labs
        return parametres

    # calculat linear forward
    def __linear_forward(self,A, W, B):
        Z = np.dot(W, A) + B
        return Z

    # Activation functions
    def __sigmoid(self,Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def __relu(self,Z):
        A = np.maximum(0, Z)
        return A

    def __leaky_relu(self,Z, alpha=0.01):
        A = np.where(Z > 0, Z, Z * alpha)
        return A

    def __tanh(self,Z):
        A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return A

    # softmax function
    def __softmax(self,Z):
        t = np.exp(Z)
        s = sum(t)
        soft = t / s
        return soft

    #calcule linear activation forward
    def __linear_activation_forward(self,input, parametres, activation):
        A = input
        N = len(parametres) / 2
        cache_z = {}
        cache_a = {}
        for i in range(1, int(N) + 1):
            W = parametres["W" + str(i)]
            B = parametres["B" + str(i)]
            Z = self.linear_forward(A, W, B)
            cache_z["Z" + str(i)] = Z
            if activation[i - 1] == "relu":
                A = self.relu(Z)
            elif activation[i - 1] == "tanh":
                A = self.tanh(Z)
            elif activation[i - 1] == "leaky_relu":
                A = self.leaky_relu(Z)
            elif activation[i - 1] == "sigmoid":
                A = self.sigmoid(Z)
            else:
                A = self.softmax(Z)
            cache_a["A" + str(i)] = A
        return A, cache_z, cache_a

    # compute loss function
    def __Loss_function(self,output, y):
        m = output.shape[1]
        l = y * np.log(output)
        loss = (sum(-sum(l).reshape(m, 1)))
        return loss / m

    # derivative of activation function
    def __D_sigmoid(self,Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def __D_relu(self,Z):
        return np.where(Z >= 0, 1, 0)

    def __D_leaky_relu(self,Z, alpha):
        return np.where(Z >= 0, 1, alpha)

    def __D_tanh(self,Z):
        return 1 - (self.tanh(Z) ** 2)

    # backward propagation
    def __D_Z(self,y, cache_a, cache_z, parametres, activation):
        N = len(cache_z)
        dZ_f = cache_a["A" + str(N)] - y
        dz = {"dZ" + str(N): dZ_f}
        for i in range(N, 0, -1):
            if activation[i - 1] == "sigmoid":
                dz["dZ" + str(i)] = np.multiply(np.dot(parametres["W" + str(i + 1)].T, dz["dZ" + str(i + 1)]),
                                                self.D_sigmoid(cache_z["Z" + str(i)]))
            elif activation[i - 1] == "tanh":
                dz["dZ" + str(i)] = np.multiply(np.dot(parametres["W" + str(i + 1)].T, dz["dZ" + str(i + 1)]),
                                                self.D_tanh(cache_z["Z" + str(i)]))
            elif activation[i - 1] == "relu":
                dz["dZ" + str(i)] = np.multiply(np.dot(parametres["W" + str(i + 1)].T, dz["dZ" + str(i + 1)]),
                                                self.D_relu(cache_z["Z" + str(i)]))
            elif activation[i - 1] == "leaky_relu":
                dz["dZ" + str(i)] = np.multiply(np.dot(parametres["W" + str(i + 1)].T, dz["dZ" + str(i + 1)]),
                                                self.D_leaky_relu(cache_z["Z" + str(i)]))
        return dz

    def __D_WB(self,input, cache_a, dz):
        dw = {}
        db = {}
        m = input.shape[1]
        for i in range(len(cache_a), 0, -1):
            if (i != 1):
                dw["W" + str(i)] = np.dot(dz["dZ" + str(i)], cache_a["A" + str(i - 1)].T) / m
                db["B" + str(i)] = np.sum(dz["dZ" + str(i)], axis=1, keepdims=True) / m
            else:
                dw["W" + str(i)] = np.dot(dz["dZ" + str(i)], input.T) / m
                db["B" + str(i)] = np.sum(dz["dZ" + str(i)], axis=1, keepdims=True) / m
        return dw, db

    # Update parametres
    def __Update_parametres(self,dw, db, parametres, learning_rate):
        new_parametres = {}
        for i in range(1, len(dw) + 1):
            new_parametres["W" + str(i)] = parametres["W" + str(i)] - learning_rate * dw["W" + str(i)]
            new_parametres["B" + str(i)] = parametres["B" + str(i)] - learning_rate * db["B" + str(i)]
        return new_parametres

    # training  Model
    def fit(self,input, output,labs=0.1, learning_rate=0.08, iteration=1000, show_loss=False):

        parametres = self.ini_parametres(self.hidden_dim, labs)
        loss = []
        for i in range(1, iteration):
            a, cache_z, cache_a = self.linear_activation_forward(input, parametres, self.activation_f)
            if show_loss == True and i % 10 == 0:
                print("cost :", self.Loss_function(a, output))
            loss.append(self.Loss_function(a, output))
            dz = self.D_Z(output, cache_a, cache_z, parametres, self.activation_f)
            dw, db = self.D_WB(input, cache_a, dz)
            parametres = self.Update_parametres(dw, db, parametres, learning_rate)

        self.parametres=parametres
        return np.squeeze(loss), parametres








