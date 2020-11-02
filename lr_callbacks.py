from keras.callbacks import Callback
import numpy as np
from math import exp, cos
from keras import backend as K
import matplotlib.pyplot as plt
import csv

class InternalStateHistory(Callback):

    def __init__(self, name='log.cvs'):
        super().__init__()
        self.name = name

    """
    --------------------------- F U N C T I O N   O N _ T R A I N _ B E G I N -------------------------
    """
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.lr = []
        self.maxTAccuracy = 0
        self.maxVAccuracy = 0

    def on_epoch_end(self, batch, logs={}):
        self.test_losses.append(logs.get('val_loss'))
        self.test_accs.append(logs.get('val_acc'))

        self.lr.append(K.get_value(self.model.optimizer.lr))
        self.train_losses.append(logs.get('loss'))
        self.train_accs.append(logs.get('acc'))

        if logs.get('acc') > self.maxTAccuracy:
            self.maxTAccuracy = logs.get('acc')

        if logs.get('val_acc') > self.maxVAccuracy:
            self.maxVAccuracy = logs.get('val_acc')

    def on_train_end(self, logs={}):
        self.train_losses.insert(0,'train_losses')
        self.test_losses.insert(0, 'val_losses')
        self.train_accs.insert(0, 'train_accs')
        self.test_accs.insert(0, 'val_accs')
        self.lr.insert(0, 'lr')
        with open(self.name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(self.train_losses)
            writer.writerow(self.test_losses)
            writer.writerow(self.train_accs)
            writer.writerow(self.test_accs)
            writer.writerow(self.lr)
            csvFile.close()

class Step_decay(Callback):

    def __init__(self, min_lr=1e-6, max_lr=0.1,step_size =2000, gamma=0.99994):

        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.iteration = 0
        self.expCycle = 0

    def constant_lr(self):
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def calculate_new_lr(self):
        lr = self.max_lr * exp(-self.gamma * self.iteration)
        K.set_value(self.model.optimizer.lr, max(lr, self.min_lr))

    def on_train_begin(self, logs=None):
        self.iteration = 0
        self.calculate_new_lr()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        self.calculate_new_lr()