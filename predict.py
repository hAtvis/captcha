# -*- coding: utf-8 -*-

from main import gen,decode
from keras.models import *

X, Y = next(gen(1))

model = load_model('my_model.h5')

Y_pred = model.predict(X)

print('real: %s\npred: %s'%(decode(Y), decode(Y_pred)))

