#Load necessary modules
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Load JSON file of Labelbox labels
path = "/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Github/AggregationScripts/Data/labelbox.json"
with open(path) as f:
  usfws = json.load(f)

