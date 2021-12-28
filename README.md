# OBServ_Models_Open

This repository contains the implementation of two types of pollination supply models[1]:

The folder **Lonsdorf** contains the implementation of the Lonsdorf model[2] in Google Earth Engine.

The folder **Machine Learning** contains the implementation of a machine learning pipeline to predict pollinator visitation rate, using scikit-learn in Python.

# Pipeline for Lonsdorf scores
In the following steps, it is assumed that the code
1. Copy the folders Lonsdorf\GEE\lib and Lonsdorf\GEE\models in your GEE repository

2. Upload the assets in Lonsdorf\GEE\assets to your GEE (depending on the option you choose when running the model, you might want to upload only some of the assets present in the folder)

3. Run Lonsdorf\pre-post-processing\generate_sites_predictions.py to generate a set of sites where predictions will be computed.

4. Upload generated 'csv' of sites to the assets in GEE

5. In the script Lonsdorf\GEE\models\lonsdorf, change the option 'sitesOption' to a new string, e.g. "sites_predictions" (you can use any string you want)

6. In the script Lonsdorf\GEE\lib\data, import the uploaded asset with the sites for predictions, and add a new case to the switch in the function 'getSamplingPoints'. The new case must correspond to the string used in the previous step, and must return the table with the sites for predictions

7. Run the script Lonsdorf\GEE\models\lonsdorf. It must produce a task to compute and download the Lonsdorf scores in the sites for predictions.

# Pipeline for machine learning predictions (in preparation)


References:

1) Gimenez-Garcia A, et al. (in preparation) Applicability of simple and reliable pollination supply models from local to global scale.

2) Lonsdorf E, et al. (2009) Modelling pollination services across agricultural landscapes. Annals of Botany 103(9):1589–1600.

