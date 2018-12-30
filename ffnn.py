import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import load_model
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#loads the input and output data stored in .pickle format
with open('train_input.pickle', mode='rb') as f:
    train_input = pickle.load(f)
with open('train_output.pickle', mode='rb') as f:
    train_output = pickle.load(f)
with open('test_input.pickle', mode='rb') as f:
    test_input = pickle.load(f)
with open('test_output.pickle', mode='rb') as f:
    test_output = pickle.load(f)

#creates a fc model
model = keras.Sequential([
    keras.layers.Dense(10, activation='sigmoid', input_dim=np.shape(train_input[0])[0]),
    keras.layers.Dense(200, activation='sigmoid'),
    keras.layers.Dense(50, activation='sigmoid'),
    keras.layers.Dense(200, activation='sigmoid'),
    keras.layers.Dense(np.shape(train_output[0])[0], activation='linear'),
])

#saves the model summary into txt file
with open("model.txt", mode='w') as f:
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    str_model_summary = "\n".join(stringlist)
    f.write(str_model_summary)
#saves the model into png file
plot_model(model, to_file='model.png')

#defines the lr for the optimizer
sgd = keras.optimizers.SGD(lr = 0.00001)

#sets the loss function and the optimizer for the model
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'],)

#trains the model and outputs the history to a variable
history = model.fit(train_input, train_output,
                    #initial_epoch = 299,
                    epochs=3000,
                    batch_size=512,
                    validation_data=(test_input,test_output),
                    verbose = 2)

#evalutes the training result
test_loss, test_acc = model.evaluate(test_input, test_output)
print('Test loss:', test_loss, 'Test accuracy:', test_acc)

#saves the model into hdf5 format
model.save('my_model.h5')

#plots the loss and val_loss
plt.xlabel("epochs")
plt.ylabel("loss")
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val_loss")
plt.legend(loc = 'upper right')
plt.title('loss and validation_loss')

#saves the plot into png and displays the plot
plt.savefig('train.png', pad_inches=0.1) #pad_inches is used to prevent the output png from having excess padding
plt.show()