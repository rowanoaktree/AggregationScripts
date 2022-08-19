import os
from deepforest import main
from deepforest import get_data
from deepforest import evaluate, visualize
import rasterio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils.prec_rec_curve import precRecCurve, save_precrec_plot

dir = "modelstates"
model = main.deepforest.load_from_checkpoint("{}/checkpoint.pl".format(dir))

#model = main.deepforest()
#model.use_bird_release()

#Predict a single image
#image_path = get_data("example.png")
#boxes = model.predict_image(path=image_path, return_plot = False)

#Predict a raster tiles
#raster_path = get_data("example.tif")
# Window size of 300px with an overlap of 25% among windows for this small tile.
#predicted_raster = model.predict_tile(raster_path, return_plot = True, patch_size=300,patch_overlap=0.25)

bbox_max_area = 10000

#Predict a set of images
csv_file = r"/datadrive/Data/val_filter4.csv" #csv file of test annotations specifying the path 
root_dir = r"/datadrive/Data/val"
save_dir = r"predictions/val"
os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'evaluation'), exist_ok=True)
predictions = model.predict_file(csv_file=get_data(csv_file), root_dir=root_dir)        #,savedir=os.path.join(save_dir, 'predictions'))

# filter too large bboxes & plot predictions
areas = (predictions['xmax'] - predictions['xmin']) * (predictions['ymax'] - predictions['ymin'])
predictions = predictions[areas <= bbox_max_area]
visualize.plot_prediction_dataframe(predictions, root_dir, None, savedir=os.path.join(save_dir, 'predictions'))
predictions.to_csv(os.path.join(save_dir, 'predictions.csv'))

#Evaluate
result = evaluate.evaluate(predictions=predictions, ground_df=pd.read_csv(csv_file), root_dir=root_dir, savedir=os.path.join(save_dir, 'evaluation'))
result.to_csv(os.path.join(save_dir, 'results.csv'))

# precision-recall curves
precRec = precRecCurve(predictions, result['results'], num_steps=50)
save_precrec_plot(precRec, os.path.join(save_dir, 'prec_rec.png'))

# class-agnostic
precRec = precRecCurve(predictions, result['results'], num_steps=50, agnostic_label='bird')
save_precrec_plot(precRec, os.path.join(save_dir, 'prec_rec_classAgnostic.png'))


#ap = result["box_precision"].mean()
#ar = result["box_recall"].mean()
#perclass = result["class_recall"]

#print("The average precision is {}, the average recall is {}".format(ap,ar))