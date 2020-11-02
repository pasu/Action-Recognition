A simple intruduction:

First, in the current folder, we have cnn_A.py, cnn_B.py, cnn_combinedModel.py, cnn_model.py, cnn_predict.py, generator.py, lr_callbacks.py, myconfig.py, totally 8 code files.

myconfig.py: global configurations
generator.py: image manager, such as creating generator for training, prediction and evaluation
cnn_model.py: model manager, creating one model for training or prediction
lr_callbacks.py: callback of model.fit_generator including InternalStateHistory and Step_decay

cnn_A.py: used to train a model
cnn_predict.py: used to evaluate the accuracy of one model
cnn_combinedModel.py: used to optmize the transfer learning model automatically
cnn_B.py: statistics of the results, such as accuracy and plots


Task A:

First, copy Stanford40 dataset in the current folder including ImageSplits, JPEGImages, XMLAnnotations (we do not use the bndbox attribute)

Second, check the variables in myconfig.py such as batch_size, nb_epoch


Finally, in cnn_A.py, select the name of the model you want to train, and run this file

Task B:
Learning Rate:
in cnn_A.py, create lr_callbacks.Step_decay, then run this file to train the model

Weight Decay:
First, in myconfig.py, set the value of WEIGHT_DECAY
Second, in cnn_A.py, select the name of the model you want to train, and run this file

Transfer Learning:
in cnn_A.py, set name = 'mobilenet', then, you can run this file to train the mobile net model

Task C:

run cnn_combinedModel file to optimize transfer learning mobilenet


