import os

# Directories
models_repo = "C:/Users/Angel/git/Observ_models"
field_repo = "C:/Users/Angel/git/OBservData"
dir_root = os.path.join(models_repo, 'code', 'Prepare_Data')
dir_features = os.path.join(models_repo, 'data', 'GEE', 'GEE features')
dir_field = os.path.join(field_repo, "Final_Data")
dir_log = os.path.join(dir_root, 'log')
dir_output = os.path.join(models_repo, 'data', 'Prepared Datasets')

# Files
csv_features = os.path.join(dir_features, 'Features.csv')
csv_field = os.path.join(dir_field, 'CropPol_field_level_data.csv')
