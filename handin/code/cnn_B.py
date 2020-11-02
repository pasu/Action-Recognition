import numpy as np
np.random.seed(1337) # for reproducibility
import generator
import myconfig
import  cnn_model
import matplotlib.pyplot as plt
import csv
from keras.callbacks import EarlyStopping
import  lr_callbacks
from keras import backend as K
import gc

# weight decay test
def gen_weight_decay():
    data_info = generator.gen_data_augmentation(myconfig.batch_size)

    list = [0, 0.0001, 0.0005, 0.01, 0.1, 1]

    for i in list:
        myconfig.WEIGHT_DECAY = i

        model = cnn_model.model_visualization(myconfig.nb_classes, mode='alex_8')

        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
        history = lr_callbacks.InternalStateHistory(str(i) + '.csv')
        learningrate = lr_callbacks.Step_decay(min_lr=0.001, max_lr=0.01, step_size=15, gamma=0.015)

        model.fit_generator(data_info[0],
                      steps_per_epoch=int(data_info[2] // myconfig.batch_size),
                      epochs=myconfig.nb_epoch,
                      validation_data=data_info[1],
                      validation_steps=int(data_info[3] // myconfig.batch_size),
                      callbacks=[history, learningrate, early],
                      verbose=2)

        del model
        gc.collect()

        K.clear_session()

def _plot_graph(data_list, label_list, xlabel, ylabel, ymin=0, ymax=0.35):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(data_list)):
        ax.plot(np.arange(len(data_list[i])), data_list[i], label=label_list[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.ylim((ymin,ymax))
    plt.show()

def _plot_bar(data_list, label_list, xlabel, ylabel, ymin=0, ymax=0.35):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(data_list)):
        ax.bar(np.arange(len(data_list[i])), data_list[i], label=label_list[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.ylim((ymin,ymax))
    plt.show()

def draw_weight_decay():
    data_list = []
    label_list = []

    list = [0, 0.0001, 0.0005, 0.01, 0.1, 1]

    for i in list:
        accuracy1 = []
        with open(str(i)+'.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            idx = 0
            for row in readCSV:
                if idx == 6:
                    accuracy1 = row[1:]

                idx += 1

        accuracy1 = [float(i) for i in accuracy1]
        accuracy1 = np.asarray(accuracy1)

        data_list.append(accuracy1)
        label_list.append(str(i))

    _plot_graph(data_list, label_list,'Epoch', 'Accuracy', 0, 0.25)

def draw_lr():
    data_list = []
    label_list = []

    accuracy1_t = []
    accuracy1 = []
    with open('a_8_1_ncrop.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in readCSV:
            if idx == 4:
                accuracy1_t = row[1:]
            if idx == 6:
                accuracy1 = row[1:]

            idx += 1

    accuracy1_t = [float(i) for i in accuracy1_t]
    accuracy1 = [float(i) for i in accuracy1]

    accuracy1_t = np.asarray(accuracy1_t)
    accuracy1 = np.asarray(accuracy1)

    data_list.append(accuracy1)
    data_list.append(accuracy1_t)

    label_list.append('val data+exponent lr')
    label_list.append('train data+exponent lr')

    accuracy2_t = []
    accuracy2 = []
    with open('a_8_1_ncrop_nolr.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in readCSV:
            if idx == 4:
                accuracy2_t = row[1:]

            if idx == 6:
                accuracy2 = row[1:]

            idx += 1

    accuracy2_t = [float(i) for i in accuracy2_t]
    accuracy2 = [float(i) for i in accuracy2]

    accuracy2_t = np.asarray(accuracy2_t)
    accuracy2 = np.asarray(accuracy2)

    data_list.append(accuracy2)
    data_list.append(accuracy2_t)

    label_list.append('val data+constant lr')
    label_list.append('train data+constant lr')

    _plot_graph(data_list, label_list, 'Epoch', 'Accuracy', 0, 0.45)

def draw_bar():
    data_list = []
    label_list = []

    accuracy1_t = []
    with open('a_30_common.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in readCSV:
            if idx == 2:
                accuracy1_t = row

            idx += 1

    accuracy1_t = [float(i) for i in accuracy1_t]

    accuracy1_t = np.asarray(accuracy1_t)

    data_list.append(accuracy1_t)

    label_list.append('Our model')

    accuracy2_t = []
    with open('b_60_tl.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in readCSV:
            if idx == 2:
                accuracy2_t = row
            idx += 1

    accuracy2_t = [float(i) for i in accuracy2_t]

    accuracy2_t = np.asarray(accuracy2_t)

    data_list.append(accuracy2_t)

    label_list.append('Transfer learning')

    _plot_bar(data_list, label_list, 'Epoch', 'Accuracy', 0, 0.45)

#draw_lr()
# first generate weight decay log
gen_weight_decay()
# second draw the accuracy curve
draw_weight_decay()