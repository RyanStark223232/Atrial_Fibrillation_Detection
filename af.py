import numpy as np
import scipy.io
from csv import reader
import keras as k
import keras.layers as l
import keras.backend as K

dropout = 0.2
fold = 5
input_size = 18286

def one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def load_padded_af (folds, norm = True):
    data_size = 8528
    data_min = -10636.
    data_max = 8318.
    input_size = 18286
    index_set = list(range(data_size))
    data = []
    for i in range(data_size):
        mat = scipy.io.loadmat('training2017/A{:05}.mat'.format(i+1))
        mat = mat['val'][0]
        if norm:
            mat = (mat - data_min)/(data_max - data_min) - 0.5
        mat = np.concatenate([mat, np.zeros([input_size - len(mat)])], 0)
        data.append(mat)        
    data = np.array(data)
    y_index = np.random.choice(index_set, [data_size//folds], False)
    x_index = np.array([i for i in index_set if i not in y_index])
    label = []
    label_mean = {'N':0, 'A':1, 'O':2, '~':3}
    with open('training2017/REFERENCE.csv') as file:
        r = reader(file)
        for i in r:
            label.append(label_mean[i[1]])
    label = np.array(label)
    train_x = data[x_index]
    train_y = one_hot(label[x_index], 4)
    test_x = data[y_index]
    test_y = one_hot(label[y_index], 4)
    return (train_x, train_y, test_x, test_y)

train_x, train_y, test_x, test_y = load_padded_af(fold)
K.clear_session()

def res_block(x, size):
    x_shortcut = x
    x = l.Dense(size)(x)
    x = l.BatchNormalization()(x)
    x = l.LeakyReLU()(x)
    x = l.Dense(size)(x)
    x = l.BatchNormalization()(x)
    x = l.LeakyReLU()(x)
    x = l.Dense(size)(x)
    x = l.BatchNormalization()(x)
    x = l.Add()([x, x_shortcut])
    x = l.LeakyReLU()(x)
    return x

def down_scale(x, size):
    x = l.Dense(size)(x)
    x = l.BatchNormalization()(x)
    x = l.LeakyReLU()(x)
    x = l.Dropout(rate = dropout)(x)
    return x

x_input = k.Input([input_size])   
x = down_scale(x_input, 1024)
x = res_block(x, 1024)
x = down_scale(x, 512)
x = res_block(x, 512)
x = down_scale(x, 256)
x = res_block(x, 256)
x = down_scale(x, 4)
x = l.Dense(4, activation = 'softmax')(x)
model = k.Model(inputs = x_input, outputs = x)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs = 1000, batch_size = 8)
preds = model.evaluate(test_x, test_y, batch_size = 32)
print("LOSS:{} ACC:{}".format(preds[0], preds[1]))