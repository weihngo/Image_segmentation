"""
Generate the test image file list (test.lst) for the prediction pipeline.

This script scans a directory of satellite remote-sensing images and writes
the absolute path of every image file into ``test.lst``, one path per line.
The resulting list file is consumed by ``predict.py``, which reads the paths
to load test images for inference.

Other ``.lst`` files in this directory (train.lst, val.lst, trainval.lst)
serve a similar purpose for training and evaluation (see ``train.py`` and
``eval.py``).  They are loaded by ``utils/basic_dataset.py`` (BasicDataset)
to build PyTorch datasets.

Usage:
    cd data/
    python gen_list.py

Note:
    * ``data_dir`` should point to the directory that holds the raw test
      images relative to this script's location.
    * The absolute prefix written into ``test.lst`` must match the path
      where the images reside on the machine that runs prediction.
      Adjust both ``data_dir`` and the prefix string as needed for your
      environment.
"""

import os

# Relative path from this script to the directory containing test images.
data_dir = "../../data/image_A"

# Create (or overwrite) the test image list file.
f = open("test.lst", 'w')

# Iterate over every file in the image directory and write its absolute path.
for i in os.listdir(data_dir):
    f.write("/home/archlab/lzr_satellite_image_regonization/data/image_A/" + str(i) + '\n')

f.close()
