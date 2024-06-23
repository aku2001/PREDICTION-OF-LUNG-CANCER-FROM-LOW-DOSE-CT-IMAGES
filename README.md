# PREDICTION OF LUNG CANCER FROM LOW-DOSE CT IMAGES

This Project aims to provide an effective solution to detection of early-stage lung
cancer from low-dose CT scans. Comparing to high dose CT scans, low dose CT
scans provides less risk and lower radiation exposure. However, it also decreases the
scan quality and its readability. Therefore, preprocessing methods and deep learning
techniques are used to enhance the scan quality and detect the nodules that might
turn into cancer. Specifically engineered 3D CNN is utilized for the detection of the
lung nodule. The model successfully detects the nodule location and classifies it as a
nodule.

A web-based UI is also developed. It gets the ct scans as input and provides an
enhanced image and detection result. The ct scans gotten from the UI sends back
to the server. The server preprocess the image and provides a detection result that is
gotten from the model. The result is sent back to the UI so the doctors can make sense
of it and provides needed medical treatment to the patient.


For more detail please read the article.

## How To Run

To preprocess the mhd file run the Preprocess_Main Notebook. It applies preprocessing and augmentation at the same time.
It is tested against Luna and Elcap dataset. For the Elcap dataset it should be converted to mhd format from dicom format. This can be achived by running Dicom_Mhd_Converter Notebook.
If the Elcap dataset is used in the preprocess notebook, resampling and change coordinates preprocessing steps should be skipped. 
It is becuase of the annotations given in voxel coordinates not world coordinates.

After preprocessing Model Training Notebook should be used to train the model. After the traingin and creation of the model. The model can be used with the Web UI. 
In the Web UI code "app.py" change the path for the model with the trained model. Then run the app.py. In the web UI the preprocessing step is applied automatically. 
It receives a zip file containing mhd and raw file.