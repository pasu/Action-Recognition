'''
   file name: model.py
   create alexnet model based on keras
   create alexnet_8 model based on keras, only 8 layers before flatten operator
   create mobilenet model for transfer learning
'''

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.applications import MobileNet
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import myconfig

# add conv2d layer
def add_conv2d(x, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
               activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
               bias_constraint=None, weight_decay=myconfig.WEIGHT_DECAY):

    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
              dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
              activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
              bias_constraint=bias_constraint)(x)

    return layer

def create_MobileNet(nb_class):
    model_vgg = MobileNet(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3))

    for layer in model_vgg.layers:
        layer.trainable = False

    x = model_vgg.output
    x = Flatten()(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(myconfig.DROPOUT)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(myconfig.DROPOUT)(x)

    x = Dense(nb_class, activation="softmax")(x)

    return  x, model_vgg.input

def create_alexnet_8_1(nb_class):
    _input = Input((myconfig.img_size, myconfig.img_size, myconfig.img_channel))

    nScale = 4

    # convolution layer 1
    conv1 = add_conv2d(_input, int(96 / nScale), (11, 11), strides=(4, 4), padding='same')
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(act1)

    # convolution layer 2
    conv2 = add_conv2d(pool1, int(256 / nScale), (5, 5), (1, 1), padding='same')
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(act2)

    # convolution layer 3
    conv3 = add_conv2d(pool2, int(384 / nScale), (3, 3), (1, 1), padding='same')
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    flat1 = Flatten()(act3)

    # fc Layer 6
    dense1 = Dense(int(4096 / nScale))(flat1)
    bn6 = BatchNormalization()(dense1)
    act6 = Activation('relu')(bn6)
    drop1 = Dropout(myconfig.DROPOUT)(act6)

    # fc Layer 7
    dense2 = Dense(int(4096 / nScale))(drop1)
    bn7 = BatchNormalization()(dense2)
    act7 = Activation('relu')(bn7)
    drop2 = Dropout(myconfig.DROPOUT)(act7)

    # output Layer
    x = Dense(output_dim=nb_class)(drop2)
    bn8 = BatchNormalization()(x)
    act8 = Activation('softmax')(bn8)

    return act8, _input

def create_alexnet(nb_class):
    _input = Input((myconfig.img_size, myconfig.img_size, myconfig.img_channel))

    nScale = 6

    # convolution layer 1
    conv1 = add_conv2d(_input, int(96 / nScale), (11, 11), strides=(4, 4), padding='same')
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(act1)

    # convolution layer 2
    conv2 = add_conv2d(pool1, int(256 / nScale), (5, 5), (1, 1), padding='same')
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(act2)

    # convolution layer 3
    conv3 = add_conv2d(pool2, int(384 / nScale), (3, 3), (1, 1), padding='same')
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    # convolution layer 4
    conv4 = add_conv2d(act3, int(384 / nScale), (3, 3), (1, 1), padding='same')
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)

    # convolution layer 5
    conv5 = add_conv2d(act4, int(256 / nScale), (3, 3), (1, 1), padding='same')
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(bn5)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(act5)

    flat1 = Flatten()(pool3)

    # fc Layer 6
    dense1 = Dense(int(4096/ nScale ))(flat1)
    bn6 = BatchNormalization()(dense1)
    act6 = Activation('relu')(bn6)
    drop1 = Dropout(myconfig.DROPOUT)(act6)

    # fc Layer 7
    dense2 = Dense(int(4096/ nScale))(drop1)
    bn7 = BatchNormalization()(dense2)
    act7 = Activation('relu')(bn7)
    drop2 = Dropout(myconfig.DROPOUT)(act7)

    # output Layer
    x = Dense(output_dim=nb_class)(drop2)
    bn8 = BatchNormalization()(x)
    act8 = Activation('softmax')(bn8)

    return  act8, _input

def model_visualization(nb_class=40, mode='alex'):
    # Create the Model
    if mode == 'alex':
        img_model, img_input = create_alexnet(nb_class)
    if mode == 'alex_8':
        img_model, img_input = create_alexnet_8_1(nb_class)
    elif mode == 'mobilenet':
        img_model, img_input = create_MobileNet(nb_class)

    # Create a Keras Model
    model=Model(input=img_input,output=[img_model])

    model.summary()

    # Save a PNG of the Model Build
    plot_model(model, to_file=mode + '.png', show_shapes=True)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

