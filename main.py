# -*- coding: utf-8 -*-


import numpy as np
from captcha.image import ImageCaptcha
import random
import matplotlib.pyplot as plt
from settings import *

def gen(batch_size=32):
    X = np.zeros((batch_size, HEIGHT, WIDTH, 3), dtype=np.uint8)
    Y = [np.zeros((batch_size, N_CLASS), dtype=np.uint8) for i in range(N_LENGTH)]
    generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
    while True:
        for i in range(batch_size):
            random_str = "".join([random.choice(characters) for k in range(N_LENGTH)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                Y[j][i,:] = 0
                Y[j][i,characters.find(ch)] = 1
        yield X, Y

def decode(y):
    y = np.argmax(y, axis=2)[:,0]
    return ''.join([characters[x] for x in y])

if __name__ == "__main__":
    X, Y = next(gen(1))
    plt.imshow(X[0])
    print(decode(Y))
