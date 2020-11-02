import gc
import random
import cv2, os
import xml.etree.ElementTree as ET
import numpy as np
import myconfig
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications import MobileNet
import shutil, os

xml_folder = './XMLAnnotations/'
img_folder = './JPEGImages/'
split_folder = './ImageSplits/'

img_label = []
action_count = []
unique_list = []

# iterate the subfolder and all files in this directory
# then split the images into test and validation parts by train_test_split
# directory: the directory path
# return: path name of test data and validation data, index of test data and validation data
def paths_list_from_directory():
    img_train = []
    img_test = []

    with open(split_folder+'train.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    for i in content:
        (shortname, extension) = os.path.splitext(i)
        img_train.append(shortname + '.xml')


    with open(split_folder+'test.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    for i in content:
        (shortname, extension) = os.path.splitext(i)
        img_test.append(shortname + '.xml')

    img_paths = os.listdir(xml_folder)

    for i in img_paths:
        tree = ET.parse(xml_folder + i)
        root = tree.getroot()
        obj = root.find('object')
        action_type = obj.find('action').text
        img_label.append(action_type)

    u_list = list(set(img_label))
    # set is unordered list, may have label mismatch
    u_list.sort()

    return [img_train, img_test, u_list]

image_cache = {}

def qload_image(filename):
    value = image_cache.get(filename)
    if value != None:
        return value

    # step 1: parse xml annotation
    tree = ET.parse(filename)
    root = tree.getroot()
    img_name = img_folder + root.find('filename').text
    obj = root.find('object')
    action_type = obj.find('action').text
    bndbox = obj.find('bndbox')
    xStart = int(bndbox.find('xmin').text)
    xEnd = int(bndbox.find('xmax').text)
    yStart = int(bndbox.find('ymin').text)
    yEnd = int(bndbox.find('ymax').text)

    # step 2: append the label of this image
    index = unique_list.index(action_type)
    label = np.eye(myconfig.nb_classes, dtype=np.uint8)[index]

    # step 3: crop the correct region of this image as the image_data
    if myconfig.img_channel == 1:
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(img_name)

    image_cache[filename] = (image, label)
    return (image, label)

def resize_with_pad(image, height=myconfig.img_size, width=myconfig.img_size):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image

# load image from the file and crop it to 48*48 and save the image data with its label(cat or dog)
# filename: the path of the image
# return: the image data and its label
def load_image(filename):
    value = image_cache.get(filename)
    if value != None:
        return value

    # step 1: parse xml annotation
    tree = ET.parse(filename)
    root = tree.getroot()
    img_name = img_folder + root.find('filename').text
    obj = root.find('object')
    action_type = obj.find('action').text
    bndbox = obj.find('bndbox')

    # step 2: append the label of this image
    index = unique_list.index(action_type)
    label = np.eye(myconfig.nb_classes, dtype=np.uint8)[index]

    # step 3: crop the correct region of this image as the image_data
    if myconfig.img_channel == 1:
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(img_name)

    image_data = image

    image_data = resize_with_pad(image_data);

    if myconfig.bScale == True:
        image_data = image_data.astype('float32')
        image_data = np.multiply(image_data, 1.0 / 255.0)

    image_data = np.reshape(image_data,(myconfig.img_size, myconfig.img_size, myconfig.img_channel))

    image_cache[filename] = (image_data, label)
    return (image_data, label)

def DataGenerator(img_addrs, batch_size):
  while 1:
    # Ensure randomisation per epoch
    random.shuffle(img_addrs)

    X = []
    Y = []

    n = np.floor((len(img_addrs)-1)/batch_size).astype(int)
    for i in range(n):
        batch_data = [load_image(xml_folder + img_addrs[i*batch_size+j]) for j in range(batch_size)]
        X = [d[0] for d in batch_data]
        Y = [d[1] for d in batch_data]

        image = np.array(X)

        label = np.array(Y, dtype=np.uint8)
        label = label.reshape(len(Y), myconfig.nb_classes)
        yield (image, label)
        del X, Y, image, label
        gc.collect()
        X = []
        Y = []

def split_data():
    paths = paths_list_from_directory()

    global unique_list
    unique_list = paths[2]

    X_train = paths[0]
    X_val = paths[1]

    img_test = []
    for f in X_val:
        (shortname, extension) = os.path.splitext(f)
        path = shortname + '.jpg'
        img_test.append('./Test/' + path)
        shutil.copy(img_folder + path, './Test/')

    img_train = []
    for f in X_train:
        (shortname, extension) = os.path.splitext(f)
        path = shortname + '.jpg'
        img_train.append('./Train/' + path)
        shutil.copy(img_folder + path, './Train/')

def getImgWithLabels(imglist):
    X_list = []
    Y_list = []
    for i in imglist:
        X, Y = load_image(xml_folder + i)
        X_list.append(X)
        Y_list.append(Y)

    return  (X_list, Y_list)

def gen_cache():
    paths = paths_list_from_directory()
    global unique_list
    unique_list = paths[2]

    x_train, y_train = getImgWithLabels(paths[0])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_test, y_test = getImgWithLabels(paths[1])
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    np.save('train_data.npy', x_train)
    print('generate train_data.npy')
    np.save('train_label.npy', y_train)
    print('generate train_label.npy')
    np.save('test_data.npy', x_test)
    print('generate test_data.npy')
    np.save('test_label.npy', y_test)
    print('generate test_label.npy')


def gen_data_augmentation(batch_size=32):
    if myconfig.support_cacheImg == True:
        if os.path.isfile('./train_data.npy') == False:
            gen_cache()

        x_train = np.load('train_data.npy')
        y_train = np.load('train_label.npy')
        x_test = np.load('test_data.npy')
        y_test = np.load('test_label.npy')
    else:
        paths = paths_list_from_directory()
        global unique_list
        unique_list = paths[2]

        x_train, y_train = getImgWithLabels(paths[0])
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        x_test, y_test = getImgWithLabels(paths[1])
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

    train_gen = ImageDataGenerator(featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    rotation_range=myconfig.rotation_range,
                                    width_shift_range=myconfig.width_shift_range,
                                    height_shift_range=myconfig.height_shift_range,
                                    horizontal_flip=myconfig.horizontal_flip,
                                    zoom_range=myconfig.zoom_range,
                                    vertical_flip=False)

    train_gen.fit(x_train)
    train_gen_generator = train_gen.flow(x_train, y_train, batch_size=myconfig.batch_size)

    test_gen = ImageDataGenerator()
    test_gen_generator = test_gen.flow(x_test, y_test, batch_size=myconfig.batch_size)

    train = len(x_test)
    test = len(x_train)

    return  [train_gen_generator, test_gen_generator, train, test]

def hard_negative_mining(model, img_addrs, batch_size=32):
    while 1:
        # Ensure randomisation per epoch
        random.shuffle(img_addrs)

        X = []
        Y = []

        batch_id = 0
        n = np.floor((len(img_addrs) - 1) / batch_size).astype(int)
        while batch_id < n:
            while len(X)<batch_size and batch_id < n:
                batch_data = [load_image(xml_folder + img_addrs[batch_id * batch_size + j]) for j in range(batch_size)]
                X_data = [d[0] for d in batch_data]
                Y_data = [d[1] for d in batch_data]

                data = []
                data.append(X_data[0])
                data = np.asarray(data)
                predict = model.predict(data)

                image = np.array(X_data)
                predict = model.predict(image)
                errors = np.abs(predict - Y_data).max(axis=-1) > .99
                X += X_data[errors].tolist()
                Y += Y_data[errors].tolist()
                batch_id += 1

            image = np.array(X_data)
            label = np.array(Y, dtype=np.uint8)
            label = label.reshape(len(Y), myconfig.nb_classes)
            yield (image, label)
            del X, Y, image, label
            gc.collect()
            X = []
            Y = []

def _evaluate():
    if myconfig.support_cacheImg == True:
        x_test = np.load('test_data.npy')
        y_test = np.load('test_label.npy')
    else:
        paths = paths_list_from_directory()
        global unique_list
        unique_list = paths[2]

        x_test, y_test = getImgWithLabels(paths[1])
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

    return  (x_test, y_test)

def evaluate_data(batch_size=32):
    x_test, y_test = _evaluate()

    test_gen = ImageDataGenerator()
    test_gen_generator = test_gen.flow(x_test, y_test, batch_size=myconfig.batch_size)

    return  [test_gen_generator, len(x_test)]

def evaluate_data2(batch_size=32):
    x_test, y_test = _evaluate()

    return  [x_test, y_test, len(x_test)]

def gen_data(batch_size=32):
    paths = paths_list_from_directory()

    global unique_list
    unique_list = paths[2]

    X_train = paths[0]
    X_val = paths[1]

    train_gen = DataGenerator(X_train, batch_size)
    test_gen = DataGenerator(X_val, batch_size)

    return [train_gen,test_gen,len(X_train),len(X_val)]

def gen_tl():
    model_vgg = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3))

    x = model_vgg.output

    for layer in model_vgg.layers:
        layer.trainable = False

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model_vgg.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    x_test = np.load('train_data.npy')
    i_list = []
    for i in x_test:
        data = []
        data.append(i)
        data = np.asarray(data)
        preds = model_vgg.predict(data)
        i_list.append(preds[0])

    x_test = np.asarray(i_list)
    np.save('tf_output_train.npy', x_test)
    print('generate tf_output_train.npy')

    x_test = np.load('test_data.npy')
    i_list = []
    for i in x_test:
        data = []
        data.append(i)
        data = np.asarray(data)
        preds = model_vgg.predict(data)
        i_list.append(preds[0])

    x_test = np.asarray(i_list)
    np.save('tf_output_test.npy', x_test)
    print('generate tf_output_test.npy')