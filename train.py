import datetime
from model.model import UNet_seg
import tensorflow as tf
import os
from data_loader import Data_Loader
from utils import print_cost_time

start_time = datetime.datetime.now()
load_weights = False
checkpoint_save_path = './checkpoint/demo1.ckpt'
batch_size = 8
epochs = 0

#   load_file_mode部分数据为part 便于测试 全部数据为all 其实也可以随便写 if part else all
data_loader = Data_Loader(load_file_mode='all', mask_size=256, rewrite_hdf5=False)

train_img, train_label = data_loader.load_train_data()
val_img, val_label = data_loader.load_val_data()

model = UNet_seg(filters=32, img_width=512, input_channel=3, num_class=13, num_con_unit=2)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

if os.path.exists(checkpoint_save_path+'.index') and load_weights:
    print("[INFO] loading weights")
    model.load_weights(checkpoint_save_path)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    mode='auto',
    save_freq='epoch'
)

history = model.fit(
    train_img, train_label, batch_size=batch_size, epochs=epochs,
    validation_data=(val_img, val_label), validation_freq=1,
    callbacks=[checkpoint_callback]
)

model.summary()

print_cost_time(start_time)
