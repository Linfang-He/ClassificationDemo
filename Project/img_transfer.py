# coding:utf-8

from PIL import Image
import numpy as np


def loadImage(path):
    # load image
    im = Image.open(path).resize((28,28))
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data)

    # transform shape to 28 * 28
    new_data = np.reshape(data, (28,28))

    # save transformed image
    new_im = Image.fromarray(new_data)
    new_im = new_im.convert('RGB')
    new_im.save(path)

    return new_data