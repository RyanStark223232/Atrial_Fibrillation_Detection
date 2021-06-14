import numpy as np
from csv import reader
import tensorflow as tf
from tensorflow import keras as k

dropout = 0.2
fold = 5
input_size = 5000

def one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def load_sliced_af (folds, norm = True, over = True):
    data_size = 8528
    data_min = -10636.
    data_max = 8318.
    data = []
    label = []
    label_mean = {'N':0, 'A':1, 'O':2, '~':3}
    with open('heartbeats_data.csv') as file:
        r = reader(file)
        for i in r:
            mat = np.array([int(j) for j in i[2:]])        
            if norm:
                mat = (mat - data_min)/(data_max - data_min) - 0.5
            while len(mat) > input_size:
                temp = mat[:input_size]
                mat = mat[input_size:]
                data.append(temp)
                label.append(label_mean[i[1]])
            mat = np.concatenate([mat, np.zeros([input_size - len(mat)])], 0)
            data.append(temp)
            label.append(label_mean[i[1]])
    data = np.array(data)
    label = np.array(label)
    data_size = len(label)
    index_set = list(range(data_size))
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

train_x, train_y, test_x, test_y = load_sliced_af(fold, True)
train_x = np.expand_dims(train_x, axis = 2)
test_x = np.expand_dims(test_x, axis = 2)

k.backend.clear_session()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
k.backend.set_session(sess)

def res_block(x, size, filt):
    x_shortcut = x
    x = k.layers.Conv1D(size//4, 1, padding = 'same', kernel_regularizer=k.regularizers.l2(0.01))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Dropout(rate = dropout)(x)
    x = k.layers.Conv1D(size//4, filt, padding = 'same', kernel_regularizer=k.regularizers.l2(0.01))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Dropout(rate = dropout)(x)
    x = k.layers.Conv1D(size, 1, padding = 'same', kernel_regularizer=k.regularizers.l2(0.01))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Dropout(rate = dropout)(x)
    return x

def down_scale(x, size, filt):
    x = k.layers.Conv1D(size, filt, strides = 4, padding = 'same', kernel_regularizer=k.regularizers.l2(0.01))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.LeakyReLU()(x)
    x = k.layers.Dropout(rate = dropout)(x)
    return x

x_input = k.Input([input_size, 1])   
x = down_scale(x_input, 128, 49)
x = res_block(x, 128, 9)
x = down_scale(x, 128, 9)
x = res_block(x, 128, 9)
x = down_scale(x, 128, 9)
x = res_block(x, 128, 9)
x = down_scale(x, 128, 9)
#x = res_block(x, 256)
#x = down_scale(x, 16)
x = k.layers.Flatten()(x)
x = k.layers.Dense(4, activation = 'softmax')(x)
k.Model()
model = k.Model(inputs = x_input, outputs = x)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

result = np.expand_dims(np.array(test_y), axis = 0)
data_size = len(train_y)
index_set = list(range(data_size))
for it in range(1000):
    for it in range(1):
        bat_index_set = np.random.choice(index_set[:1024], size = 1024, replace = False)
        bat_train_x = train_x[bat_index_set]
        bat_train_y = train_y[bat_index_set]
        model.fit(bat_train_x, bat_train_y, epochs = 1, batch_size = 32)
    preds = np.argmax(model.predict(test_x), axis = 1)
    result = np.concatenate([result, np.expand_dims(preds, axis = 0)], axis = 0)
    np.save("result/conv1d_2", result)
model.save_weights("/research/ksleung5/hwong8/model.h5")