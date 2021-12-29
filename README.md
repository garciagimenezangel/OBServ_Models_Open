# OBServ_Models_Open

This repository contains the implementation of two types of pollination supply models[1]:

The folder **Lonsdorf** contains the implementation of the Lonsdorf model[2] in Google Earth Engine.

The folder **Machine Learning** contains the implementation of a machine learning pipeline to predict pollinator visitation rate, using scikit-learn in Python.

# Tips for the computation of Lonsdorf scores at arbitrary locations
1. Copy the folders Lonsdorf/GEE/lib and Lonsdorf/GEE/models in your GEE repository.

2. Upload the assets in Lonsdorf/GEE/assets to your GEE (depending on the option you choose when running the model, you might want to upload only some of the assets present in the folder). Import the assets in the GEE code editor, in the script Lonsdorf/GEE/lib/data. The names of the assets must be adapted to the ones used in Lonsdorf/GEE/lib/data (you must revise the code in Lonsdorf/GEE/lib/data to identify those names)

3. Run Lonsdorf/pre-post-processing/generate_sites_predictions.py to generate a set of sites where Lonsdorf scores will be computed.

4. Upload generated 'csv' of sites to the assets in GEE.

5. In the script Lonsdorf/GEE/lib/data, import the uploaded asset with the new sites for Lonsdorf scores, and add a new case to the switch in the function 'getSamplingPoints'. The new case must correspond to a (arbitrary, e.g. "new_sites") string that will be used in the next step in a different script. It must return the imported table with the new sites.

6. In the script Lonsdorf/GEE/models/lonsdorf, change the option 'sitesOption' to the string used in the previous step to define the new case in the switch.

7. Run the script Lonsdorf/GEE/models/lonsdorf. It must produce a task to compute and download the Lonsdorf scores at the new sites.

# Tips for machine learning predictions at arbitrary locations
1. Copy the script Machine Learning/GEE feature extraction/features in your GEE repository.

2. Follow the steps 3-5, in the tips for the computation of Lonsdorf scores (above).

3. In the script Machine Learning/GEE feature extraction/features, use the string used in Lonsdorf/GEE/lib/data to identify the new sites (e.g. "new sites"), to retrieve the table with the function data.getSamplingPoints("new sites"). 

4. Run Machine Learning/GEE feature extraction/features in GEE. You will get a csv file with the predictors at the new sites. They must be transformed in the next step before being used to compute predictions.

5. Use the script Machine Learning/ML Python/utils/process_GEE_features_into_predictors.py, to transform the data extracted in GEE, to be used to compute predictions. The transformation must be the same that is performed in the beginning of the ML pipeline for the training and test subsets, and includes standardizing numeric predictors and transforming categorical variables into numeric.

6. Use the script Machine Learning/ML Python/Regression/predict.py (see "Test 2" therein) to compute predictions, based on already trained models. 

References:

1) Gimenez-Garcia A, et al. (in preparation) Applicability of simple and reliable pollination supply models from local to global scale.

2) Lonsdorf E, et al. (2009) Modelling pollination services across agricultural landscapes. Annals of Botany 103(9):1589–1600.

