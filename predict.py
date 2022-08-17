import os
from deepforest import main
from deepforest import get_data
from deepforest import evaluate
import rasterio
from matplotlib import pyplot
import numpy as np
import pandas as pd

#dir = "modelstates"
#model = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(dir))

model = main.deepforest()
model.use_bird_release()

#Predict a single image
#image_path = get_data("example.png")
#boxes = model.predict_image(path=image_path, return_plot = False)

#Predict a raster tiles
#raster_path = get_data("example.tif")
# Window size of 300px with an overlap of 25% among windows for this small tile.
#predicted_raster = model.predict_tile(raster_path, return_plot = True, patch_size=300,patch_overlap=0.25)

#Predict a set of images
csv_file = r"/datadrive/Data/val_filter4.csv" #csv file of test annotations specifying the path 
root_dir = r"/datadrive/Data/val"
save_dir = r"predictions/val"
os.makedirs(save_dir, exist_ok=True)
predictions = model.predict_file(csv_file=get_data(csv_file), root_dir=root_dir,savedir=save_dir)

#Evaluate
result = evaluate.evaluate(predictions=predictions, ground_df=pd.read_csv(csv_file), root_dir=root_dir, savedir=save_dir)

ap = result["box_precision"].mean()
ar = result["box_recall"].mean()
perclass = result["class_recall"]

print("The average precision is {}, the average recall is {}".format(ap,ar))