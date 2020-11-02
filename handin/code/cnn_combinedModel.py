import numpy as np
np.random.seed(1337) # for reproducibility
from keras.optimizers import SGD
import myconfig
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import lr_callbacks
import csv
from keras import backend as K
import gc, os
import random
import  generator

if os.path.isfile('./train_label.npy') == False:
    generator.gen_cache()

if os.path.isfile('./tf_output_train.npy') == False:
    generator.gen_tl()

train_data = np.load('tf_output_train.npy')
train_label = np.load('train_label.npy')

train_gen = ImageDataGenerator()
train_gen_generator = train_gen.flow(train_data, train_label, batch_size=myconfig.batch_size)

test_data = np.load('tf_output_test.npy')
test_label = np.load('test_label.npy')

test_gen = ImageDataGenerator()
test_gen_generator = test_gen.flow(test_data, test_label, batch_size=myconfig.batch_size)

accuracy_cache = {}

# find the best weight when set dense1 and dense2 numbers
def best_res(dense_1_num, dense_2_num):
    if dense_1_num*10000+dense_2_num in accuracy_cache:
        return accuracy_cache[dense_1_num*10000+dense_2_num]

    _input = Input((7, 7, 1024))

    flat1 = Flatten()(_input)
    x = Dense(dense_1_num, activation="relu")(flat1)
    x = Dropout(myconfig.DROPOUT)(x)

    x = Dense(dense_2_num, activation="relu")(x)
    x = Dropout(myconfig.DROPOUT)(x)

    x = Dense(40, activation="softmax")(x)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model=Model(input=_input,output=[x])

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    name = str(dense_1_num) + '_' + str(dense_2_num) + '.h5'

    checkpoint = ModelCheckpoint(name, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    history = lr_callbacks.InternalStateHistory()
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
    model.fit_generator(train_gen_generator,
                                  steps_per_epoch=int(len(train_data) // myconfig.batch_size),
                                  epochs=myconfig.nb_epoch,
                                  validation_data=test_gen_generator,
                                  validation_steps=int(len(test_data) // myconfig.batch_size),
                                  callbacks=[checkpoint, history, early],
                                  verbose=0)

    accuracy_cache[dense_1_num * 10000 + dense_2_num] = (history.maxTAccuracy, history.maxVAccuracy)

    del model
    gc.collect()

    K.clear_session()
    return  (history.maxTAccuracy, history.maxVAccuracy)

dense1_value = myconfig.dense_start
dense2_value = myconfig.dense_start

accuracy_list = []

# optimization algorithm to find the best numbers for dense1 and dense2 within myconfig.opt_iteration
idx = 0
while idx < myconfig.opt_iteration:
    bChange2 = True

    dense2_left = dense2_value - myconfig.opt_step
    dense2_right = dense2_value + myconfig.opt_step

    result = best_res(dense1_value, dense2_value)
    
    with open("search_model.log","a") as f:
        f.write("%d,%d,%f\n"%(dense1_value,dense2_value,result[1]))

    result_left = best_res(dense1_value, dense2_left)
    result_right = best_res(dense1_value, dense2_right)

    delAccuracy1 = result_left[1] - result[1]
    delAccuracy2 = result_right[1] - result[1]

    if delAccuracy1 > 0 and delAccuracy2 > 0:
        if delAccuracy1 > delAccuracy2:
            dense2_value = dense2_left
            accuracy_list.append(result_left)
        else:
            dense2_value = dense2_right
            accuracy_list.append(result_right)
    elif delAccuracy1 > 0:
        dense2_value = dense2_left
        accuracy_list.append(result_left)
    elif delAccuracy2 > 0:
        dense2_value = dense2_right
        accuracy_list.append(result_right)
    else:
        bChange2 = False
        accuracy_list.append(result)
    

    bChange1 = True

    dense1_left = dense1_value - myconfig.opt_step
    dense1_right = dense1_value + myconfig.opt_step

    result = best_res(dense1_value, dense2_value)
    
    with open("search_model.log","a") as f:
        f.write("%d,%d,%f\n"%(dense1_value,dense2_value,result[1]))
    result_left = best_res(dense1_left, dense2_value)
    result_right = best_res(dense1_right, dense2_value)

    delAccuracy1 = result_left[1] - result[1]
    delAccuracy2 = result_right[1] - result[1]

    if delAccuracy1 > 0 and delAccuracy2 > 0:
        if delAccuracy1 > delAccuracy2:
            dense1_value = dense1_left
            accuracy_list.append(result_left)
        else:
            dense1_value = dense1_right
            accuracy_list.append(result_right)
    elif delAccuracy1 > 0:
        dense1_value = dense1_left
        accuracy_list.append(result_left)
    elif delAccuracy2 > 0:
        dense1_value = dense1_right
        accuracy_list.append(result_right)
    else:
        bChange1 = False
        accuracy_list.append(result)

    if bChange1 == False and bChange2 == False:
        dense1_value = random.randint(128+256,2048-256)
        dense2_value = random.randint(128+256,2048-256)
    if dense1_value > myconfig.dense_max or  dense1_value < myconfig.dense_min:
        break
    if dense2_value > myconfig.dense_max or  dense2_value < myconfig.dense_min:
        break

    print(dense1_value, dense2_value)

with open('opt.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(accuracy_list)
    writer.writerow([dense1_value, dense2_value])
    csvFile.close()

print(dense1_value, dense2_value)
