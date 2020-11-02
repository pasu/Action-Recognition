import numpy as np
import cnn_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
np.random.seed(1337) # for reproducibility
import lr_callbacks
import generator
import myconfig

# mode: 'alex','alex_8', 'mobilenet'
# 'alex': alex net architecture
# 'alex_8': 8-layers limitation before flatten operator
# 'mobilenet': transfer learning
name = 'alex_8'
model = cnn_model.model_visualization(myconfig.nb_classes, mode = name)
data_info = generator.gen_data_augmentation(myconfig.batch_size)

#callback function for training model
checkpoint = ModelCheckpoint(name+'.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
history = lr_callbacks.InternalStateHistory()
learningrate = lr_callbacks.Step_decay(min_lr=0.001, max_lr=0.01, step_size=15, gamma=0.015)
csv_logger = CSVLogger(name + '.log')

# training
history = model.fit_generator(data_info[0],
                              steps_per_epoch=int(data_info[2] // myconfig.batch_size),
                              epochs=myconfig.nb_epoch,
                              validation_data=data_info[1],
                              validation_steps=int(data_info[3] // myconfig.batch_size),
                              callbacks=[checkpoint, history, learningrate, early, csv_logger],
                              verbose=2)

# evaluation
data_eva = generator.evaluate_data(myconfig.batch_size)
score = model.evaluate_generator(generator=data_eva[0], steps=data_eva[1]// myconfig.batch_size, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])