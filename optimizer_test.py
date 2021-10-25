import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from optimizers import AdamAccumulation
import pandas as pd


np.random.seed(42)
random.seed(12345)
tf.random.set_seed(1234)


model = models.Sequential()

model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)

optimizer_configs = [
    {'name': 'normal_run_1', 'optimizer': Adam(lr=1e-4),
     'batch_size': 32},
    {'name': 'normal_run_2', 'optimizer': Adam(lr=1e-4),
     'batch_size': 32},
    {'name': 'accum_1_steps_run_1', 
     'optimizer': AdamAccumulation(accumulation_steps=1,
                                   learning_rate=1e-4,
                                   bias_correction=True),
     'batch_size': 32
    },
    {'name': 'accum_1_steps_run_2', 
     'optimizer': AdamAccumulation(accumulation_steps=1,
                                   learning_rate=1e-4,
                                   bias_correction=True),
     'batch_size': 32
    },
    {'name': 'accum_4_steps_run_1', 
     'optimizer': AdamAccumulation(accumulation_steps=4,
                                   learning_rate=1e-4,
                                   bias_correction=True),
     'batch_size': 8
    },
    {'name': 'accum_4_steps_run_2', 
     'optimizer': AdamAccumulation(accumulation_steps=4,
                                   learning_rate=1e-4,
                                   bias_correction=True),
     'batch_size': 8
    }
]

test_models = []
result = {}
for i in range(len(optimizer_configs)):
    test_models.append(models.clone_model(model))
    test_models[i].set_weights(model.get_weights())

for opt_config, m in zip(optimizer_configs, test_models):
    m.compile(
        optimizer=opt_config['optimizer'],
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    print('\nExp: {}'.format(opt_config['name']))
    history = m.fit(train_images, train_labels, epochs=5,
                    batch_size=opt_config['batch_size'], shuffle=False)
    result['loss_' + opt_config['name']] = history.history['loss']
    result['accuracy_' + opt_config['name']] = history.history['accuracy']


print(pd.DataFrame(result).T.sort_index())