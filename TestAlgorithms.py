
import ANN

model=ANN.ANN()
model.input(dim=10)
model.layer(dim=10,activation="tanh")
model.layer(dim=23,activation="tanh")
model.layer(dim=20,actiavtion="leaky_relu")
model.output(dim=4)
model.fit(input=None,output=None,labs=0.2,learning_rate=0.08,iteration=500,show_loss=True)


