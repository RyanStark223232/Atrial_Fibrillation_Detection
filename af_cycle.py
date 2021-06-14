from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.models import load_model
import numpy as np

image_shape = (64, 520, 1)
lambda_cycle = 10.0
lambda_id = 0.1 * lambda_cycle
optimizer = Adam(0.0002, 0.5)
epoch = None

def load_padded_af (folds, norm = True, over = True):
    data = np.load("/research/ksleung5/hwong8/{}_gan.npy".format("afx"))
    data = np.expand_dims(data, axis = 3)
    label = np.load("/research/ksleung5/hwong8/{}_gan.npy".format("afy"))
    data_size = len(label)
    index_set = list(range(data_size))
    y_index = np.random.choice(index_set, [data_size//folds], False)
    x_index = np.array([i for i in index_set if i not in y_index])
    n_x_index = [i for i in x_index if label[i] == 0]
    o_x_index = [i for i in x_index if label[i] == 2]
    n_y_index = [i for i in y_index if label[i] == 0]
    o_y_index = [i for i in y_index if label[i] == 2]
    train_A = data[n_x_index]
    train_B = data[o_x_index]
    test_A = data[n_y_index]
    test_B = data[o_y_index]
    return (train_A, train_B, test_A, test_B)
train_A, train_B, test_A, test_B = load_padded_af(3)
train_data = [train_A, train_B]
test_data = [test_A, test_B]

def define_discriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	model = Model(in_image, patch_out)
	model.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0.5), loss_weights=[0.5])
	return model

def define_generator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = BatchNormalization(axis=-1)(g)
    g_1 = g
    g = LeakyReLU(alpha=0.2)(g)    
    g = Conv2D(128, (7,7), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g_2 = g
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(256, (7,7), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2DTranspose(128, (7, 7), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Add()([g_2, g])
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2DTranspose(64, (7,7), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)    
    g = Add()([g_1, g])
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(1, (7,7), padding='same')(g)
#    out_image = BatchNormalization(axis=-1)(g)
    model = Model(in_image, g)
    return model

def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	g_model_1.trainable = True
	d_model.trainable = False
	g_model_2.trainable = False
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	output_f = g_model_2(gen1_out)
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	opt = Adam(lr=0.001, beta_1=0.5)
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

def generate_real_samples(dataset, n_samples, patch_shape):
    data_size = len(dataset)
    index_set = list(range(data_size))
    ix = np.random.choice(index_set, size = n_samples, replace = False)
    X = dataset[ix]
    y = np.ones([n_samples]+patch_shape+[1])
    return X, y

def generate_fake_samples(g_model, dataset, patch_shape):
	X = g_model.predict(dataset)
	y = np.zeros([len(X)]+patch_shape+[1])
	return X, y

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    n_epochs, n_batch, = 3000, 1
    n_patch = list(d_model_A.output_shape[1:3])
    trainA, trainB = dataset
    for i in range(n_epochs):
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        if (epoch+i+1)%10 == 0:
            temp_out = np.concatenate([X_realA, X_fakeB, X_realB, X_fakeA], axis = 0)
            np.save('/research/ksleung5/hwong8/image/out{}'.format(epoch+i+1), temp_out)
            d_model_A.save("/research/ksleung5/hwong8/model/dA.h5")
            d_model_B.save("/research/ksleung5/hwong8/model/dB.h5")
            g_model_AtoB.save("/research/ksleung5/hwong8/model/gAB.h5")
            g_model_BtoA.save("/research/ksleung5/hwong8/model/gBA.h5")
            c_model_AtoB.save("/research/ksleung5/hwong8/model/cAB.h5")
            c_model_BtoA.save("/research/ksleung5/hwong8/model/cBA.h5")        

if epoch == None:
    g_model_AtoB = define_generator(image_shape)
    g_model_BtoA = define_generator(image_shape)
    d_model_A = define_discriminator(image_shape)
    d_model_B = define_discriminator(image_shape)
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
else:
    d_model_A = ("/research/ksleung5/hwong8/model/dA.h5")
    d_model_B = ("/research/ksleung5/hwong8/model/dB.h5")
    g_model_AtoB = ("/research/ksleung5/hwong8/model/gAB.h5")
    g_model_BtoA = ("/research/ksleung5/hwong8/model/gBA.h5")
    c_model_AtoB = ("/research/ksleung5/hwong8/model/cAB.h5")
    c_model_BtoA = ("/research/ksleung5/hwong8/model/cBA.h5")
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, train_data)