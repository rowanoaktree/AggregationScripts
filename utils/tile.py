
from image_slicer import slice

import os
directory =r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Imagery/SB597/raw/"

for filename in os.listdir(directory):
    slice(filename, 56)
