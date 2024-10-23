import os
from utils import create_dir

# Base directory of the project
basedir = os.path.abspath(os.path.dirname(__file__)) 

feat_file = os.path.join(basedir, 'features_SES_age_sex.xlsx')

acoustic_measures_dir = os.path.join(basedir, 'acoustic_measures')
create_dir(acoustic_measures_dir)