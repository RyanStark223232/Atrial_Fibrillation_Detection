import numpy as np
from csv import reader
import tensorflow.keras as k
import tensorflow.keras.layers as l
import tensorflow.keras.backend as K

dropout = 0.5
fold = 5
input_size = 18300
input_last = 300

def one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def load_padded_af (folds, norm = True, over = True):
    data_size = 8528
    data_min = -10636.
    data_max = 8318.    
    index_set = list(range(data_size))
    data = []
    label = []
    label_mean = {'N':0, 'A':1, 'O':2, '~':3}
    with open('heartbeats_data.csv') as file:
        r = reader(file)
        for i in r:
            label.append(label_mean[i[1]])
            mat = [int(j) for j in i[2:]]        
            if norm:
                mat = (mat - data_min)/(data_max - data_min) - 0.5
            mat = np.concatenate([mat, np.zeros([input_size - len(mat)])], 0)
            data.append(mat)        
    data = np.array(data)
    label = np.array(label)
    y_index = np.random.choice(index_set, [data_size//folds], False)
    x_index = np.array([i for i in index_set if i not in y_index])
    counts = np.unique(label[x_index], return_counts = True)[1]
    a_x_index = [i for i in x_index if label[i] == 1]
    o_x_index = [i for i in x_index if label[i] == 2]
    x_index = list(x_index) + list(np.random.choice(a_x_index, [max(counts) - counts[1]]))
    x_index = list(x_index) + list(np.random.choice(o_x_index, [max(counts) - counts[2]]))
    train_x = data[x_index]
    train_y = one_hot(label[x_index], 4)
    test_x = data[y_index]
    test_y = label[y_index]
    return (train_x, train_y, test_x, test_y)

train_x, train_y, test_x, test_y = load_padded_af(fold, False)
train_x = np.reshape(np.expand_dims(train_x, axis = 2), [train_x.shape[0], -1, input_last])
test_x = np.reshape(np.expand_dims(test_x, axis = 2), [test_x.shape[0], -1, input_last])

K.clear_session()

model = k.Sequential()
model.add(l.Bidirectional(l.LSTM(64, return_sequences=True), input_shape=(input_size//input_last, input_last)))
model.add(l.Dropout(rate=dropout))
model.add(l.Bidirectional(l.LSTM(64)))
model.add(l.Dropout(rate=dropout))
#model.add(l.Bidirectional(l.LSTM(32)))
model.add(l.Dense(4))
model.add(l.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

result = np.expand_dims(np.array(test_y), axis = 0)
data_size = len(train_y)
index_set = list(range(data_size))
for it in range(3000):
    for it in range(10):
        bat_index_set = np.random.choice(index_set, size = 256, replace = False)
        bat_train_x = train_x[bat_index_set]
        bat_train_y = train_y[bat_index_set]
        model.fit(bat_train_x, bat_train_y, epochs = 1, batch_size = 256)
    preds = np.argmax(model.predict(test_x), axis = 1)
    result = np.concatenate([result, np.expand_dims(preds, axis = 0)], axis = 0)
    np.save("result/lstm_1", result)
model.save_weights("model.h5")