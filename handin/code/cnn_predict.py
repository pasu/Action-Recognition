import numpy as np
np.random.seed(1337) # for reproducibility
import generator
import myconfig
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import csv
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

result = generator.paths_list_from_directory()
label_list = result[2]

# the name of the model you want to evaluate, you can get them through training them
# alex_8.h5:    alex_8 model
# alex.h5:      alexnet model
# mobilenet.h5: mobile net transfer learning model
model = load_model("alex_8.h5")

def gen_confusion_matrix():
    accuracy = []

    data_list = np.load('test_data.npy')
    label = np.load('test_label.npy')

    for name in label_list:
        index = label_list.index(name)
        index = np.eye(myconfig.nb_classes, dtype=np.uint8)[index]

        item_list = []
        item_label = []

        idx = 0
        for item in label:
            if np.array_equal(item, index):
                v = data_list[idx]
                item_list.append(v)
                item_label.append(index)
            idx += 1

        cate = [0] * 40
        for i in item_list:
            data = []
            data.append(i)
            data = np.asarray(data)
            r = model.predict(data)
            r = r.tolist()[0]
            v = max(r)
            v = r.index(v)
            cate[v] += 1

        accuracy.append(cate)
    return accuracy

def draw_confusion_matrix(accuracy_list, label_list):
    df_cm = pd.DataFrame(accuracy_list, index=[i for i in label_list],
                         columns=[i for i in label_list])
    plt.figure(figsize=(10, 10))
    sn.heatmap(df_cm, annot=True)
    plt.show()

def draw_accuracy_bar():
    data = np.load('test_data.npy')
    label = np.load('test_label.npy')

    accuracy = []

    for name in label_list:
        index = label_list.index(name)
        index = np.eye(myconfig.nb_classes, dtype=np.uint8)[index]

        item_list = []
        item_label = []

        idx = 0
        for item in label:
            if np.array_equal(item, index):
                item_list.append(data[idx])
                item_label.append(index)
            idx += 1

        item_data = np.asarray(item_list)
        item_label = np.asarray(item_label)

        test_gen = ImageDataGenerator()
        test_gen_generator = test_gen.flow(item_data, item_label, batch_size=myconfig.batch_size)

        score = model.evaluate_generator(generator=test_gen_generator, steps=len(item_data) // myconfig.batch_size + 1,
                                         verbose=0)
        accuracy.append(score[1])

    with open('a_30_common.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(label_list)
        writer.writerow(accuracy)
        csvFile.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(accuracy)), np.array(accuracy), label='accuracy')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.show()

def gen_accuracy():
    data = np.load('test_data.npy')
    label = np.load('test_label.npy')

    accuracy = []

    for name in label_list:
        index = label_list.index(name)
        index = np.eye(myconfig.nb_classes, dtype=np.uint8)[index]

        item_list = []
        item_label = []

        idx = 0
        for item in label:
            if np.array_equal(item, index):
                item_list.append(data[idx])
                item_label.append(index)
            idx += 1

        item_data = np.asarray(item_list)
        item_label = np.asarray(item_label)

        test_gen = ImageDataGenerator()
        test_gen_generator = test_gen.flow(item_data, item_label, batch_size=myconfig.batch_size)

        score = model.evaluate_generator(generator=test_gen_generator, steps=len(item_data) // myconfig.batch_size + 1,
                                         verbose=0)
        accuracy.append(score[1])

    with open('class.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(label_list)
        writer.writerow(accuracy)
        csvFile.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(accuracy)), np.array(accuracy), label='accuracy')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.show()

#draw_accuracy_bar()

# accuracy of class
#gen_accuracy()

# generate confusion matrix for each category
accuracy = gen_confusion_matrix()
draw_confusion_matrix(accuracy, label_list)