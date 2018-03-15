# -*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
from settings import *

input_tensor = Input((HEIGHT, WIDTH, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(N_CLASS, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
model = Model(input=input_tensor, output = x)

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
