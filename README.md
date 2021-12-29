# OBServ_Models_Open

This repository contains the implementation of two types of pollination supply models[1]:

The folder **Lonsdorf** contains the implementation of the Lonsdorf model[2] in Google Earth Engine.

The folder **Machine Learning** contains the implementation of a machine learning pipeline to predict pollinator visitation rate, using scikit-learn in Python.

# Pipeline for the computation of Lonsdorf scores at arbitrary locations
1. Copy the folders Lonsdorf\GEE\lib and Lonsdorf\GEE\models in your GEE repository.

2. Upload the assets in Lonsdorf\GEE\assets to your GEE (depending on the option you choose when running the model, you might want to upload only some of the assets present in the folder). Import the assets in the GEE code editor, in the script Lonsdorf\GEE\lib\data. The names of the assets must be adapted to the ones used in Lonsdorf\GEE\lib\data (you must revise the code in Lonsdorf\GEE\lib\data to identify those names)

3. Run Lonsdorf\pre-post-processing\generate_sites_predictions.py to generate a set of sites where Lonsdorf scores will be computed.

4. Upload generated 'csv' of sites to the assets in GEE.

5. In the script Lonsdorf\GEE\models\lonsdorf, change the option 'sitesOption' to a new string, e.g. "new_sites" (you can use any string you want).

6. In the script Lonsdorf\GEE\lib\data, import the uploaded asset with the new sites for Lonsdorf scores, and add a new case to the switch in the function 'getSamplingPoints'. The new case must correspond to the string used in the previous step (e.g. case "new sites":), and must return the imported table with the new sites.

7. Run the script Lonsdorf\GEE\models\lonsdorf. It must produce a task to compute and download the Lonsdorf scores at the new sites.

# Pipeline for machine learning predictions at arbitrary locations (in preparation)


References:

1) Gimenez-Garcia A, et al. (in preparation) Applicability of simple and reliable pollination supply models from local to global scale.

2) Lonsdorf E, et al. (2009) Modelling pollination services across agricultural landscapes. Annals of Botany 103(9):1589–1600.

