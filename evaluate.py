import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

#Import results file

results = r"datadrive/Data/predictions/result.csv"
class_recall = results["class recall"]
n_classes = class_recall.shape[1]

##Average precision score

#calculate precision and recall
recall = results["box recall"]
precision = results["box precision"]

fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

#Per-class AP
precision = class_recall["precision"]
recall = class_recall["recall"]

for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()