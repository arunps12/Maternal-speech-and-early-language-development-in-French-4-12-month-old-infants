import os
import numpy as np

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def Hz_to_semitones(Hz_values, ref):
    return 12 * np.log2(Hz_values/ref)
