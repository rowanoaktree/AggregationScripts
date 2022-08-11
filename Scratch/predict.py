from deepforest import main
import rasterio
from matplotlib import pyplot
import numpy as np

m = main.deepforest()

m.use_bird_release()

img = "/Users/rowanconverse/Desktop/BDA_18a4_20181106_2_00295_01_04.png"

plot = m.predict_image(path=img,return_plot=True)

pyplot.figure(figsize=(30,30))
#matplotlib likes RGB colors, but DeepForest-pytorch depends on openCV which like BGR colors. To plot, invert the color order.
pyplot.imshow(plot[:,:,::-1])