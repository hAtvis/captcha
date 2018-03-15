# -*- coding: utf-8 -*-
from model import *
from main import gen

model.fit_generator(gen(), samples_per_epoch=6400, nb_epoch=1, nb_worker=2, pickle_safe=True,
        validation_data=gen(), nb_val_samples=160)

model.save('my_model.h5')

