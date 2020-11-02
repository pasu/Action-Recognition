#global variables
img_size = 224
img_channel = 3
# color /255.0
bScale = True

nb_classes = 40

batch_size = 32
nb_epoch = 150

DROPOUT=0.5
#0.0005
WEIGHT_DECAY=0.0005

# loading cache image file, which has higher performance for image processing
# requirement: 6GB momory
support_cacheImg = True

# default: 30 0.25 0.25 True 0.2
default_setting = [30, 0.25, 0.25, True, 0.2]
rotation_range=default_setting[0]
width_shift_range=default_setting[1]
height_shift_range=default_setting[2]
horizontal_flip=default_setting[3]
zoom_range=default_setting[4]

# transfer learning optimization
dense_start = 512
dense_min = 128
dense_max = 1024

opt_step = 32
opt_iteration = 20