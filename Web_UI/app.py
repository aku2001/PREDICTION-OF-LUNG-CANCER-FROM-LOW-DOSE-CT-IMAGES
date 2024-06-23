import numpy as np
import os
import time
import zipfile
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify
from preprocess import Preprocess, Augmentation
from model import ModelTest
import pandas as pd
import ast
import shutil
import matplotlib.pyplot as plt



app = Flask(__name__)


app.secret_key = 'supersecretkey'  # Needed for flashing messages
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'preprocessed'
RESULT_FOLDER = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\preprocessed\\augmented\\"
MODEL_PATH = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Models\\g_40.ckpt"
ANNOTATION_EXIST = True



if(ANNOTATION_EXIST):
    MODEL_ANNOTATION_PATH = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\preprocessed\\model_annotations.csv'
else:
    MODEL_ANNOTATION_PATH = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\preprocessed\\model_annotations_unlabeled.csv'


if ANNOTATION_EXIST:
    RESULT_FOLDER += "positives\\"
else:
    RESULT_FOLDER += "unlabeled\\"
    

images = {}
annotations = {}
model_annot = {}
preprocess_status_message = ""
filePathZip = ""


def cleanEnv():
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

    shutil.rmtree(PROCESSED_FOLDER)
    os.makedirs(PROCESSED_FOLDER)

    annot_src = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\annotations\\annotations.csv"
    annot_dest = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\uploads\\annotations.csv"
    shutil.copy(annot_src,annot_dest)

    annot_src = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\annotations\\candidates.csv"
    annot_dest = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Projeler\\Bitirme\\uploads\\candidates.csv"
    shutil.copy(annot_src,annot_dest)
    

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global preprocess_status_message, filePathZip
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('.zip'):
            cleanEnv()
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            filePathZip = filepath
            
            preprocess_status_message = "Unzipping"
            return render_template('process.html')
    return render_template('index.html')




@app.route('/process')
def process_file():
    global preprocess_status_message,filePathZip

    preprocess_status_message = "Unzipping"
    print("unzipping")
    with zipfile.ZipFile(filePathZip, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_FOLDER)

    preprocess_status_message = "Preprocessing"
    preprocess = Preprocess(resourcePath=UPLOAD_FOLDER, outputPath=PROCESSED_FOLDER,annotationExist=ANNOTATION_EXIST)
    preprocess.save_preprocessed_data()

    preprocess_status_message = "Slicing"
    augmentation = Augmentation(resourcePath=UPLOAD_FOLDER, outputPath=PROCESSED_FOLDER, annotationExist=ANNOTATION_EXIST)
    augmentation.save_augmented_data()

    preprocess_status_message = "Predicting"
    modelTest = ModelTest(outputPath=PROCESSED_FOLDER, model_path=MODEL_PATH,annotationExist=ANNOTATION_EXIST)
    modelTest.run()


    return jsonify("Finished");

@app.route('/preprocess_status')
def preprocess_status():
    global preprocess_status_message
    return jsonify(str(preprocess_status_message))    



@app.route('/result')
def result():
    # Load the images
    global images, annotations,model_annot
    image_folder = RESULT_FOLDER


    model_annotations = pd.read_csv(MODEL_ANNOTATION_PATH)
    
    
    print("image folder: ",image_folder)
    for filename in os.listdir(image_folder):
        if filename.endswith('.npy'):
            seriesuid = filename.split('_')
            sub = seriesuid[1]+"_"+seriesuid[2]+"_"+seriesuid[3].replace(".npy", "")
            seriesuid = seriesuid[0]


            filtered_annotation = model_annotations[
                    (model_annotations['seriesuid'] == seriesuid) & 
                    (model_annotations['sub_index'] == sub)
                ]
            
            parts = filename.split('_')
            i = int(parts[1])
            j = int(parts[2])
            k = int(parts[3].replace(".npy", ""))

            if (i, j, k) not in annotations:
                annotations[(i, j, k)] = [[] for _ in range(128)]  # Initialize 128 empty lists for each i, j, k
                model_annot[(i, j, k)] = [[] for _ in range(128)]
        

            if (i, j) not in images:
                images[(i, j)] = []

            images[(i, j)].append(np.load(os.path.join(image_folder, filename)))
            
            
            nodules_coords = filtered_annotation['centers'].apply(ast.literal_eval)
            for nodule_coords in nodules_coords:
                for z, y, x in nodule_coords:
                    print("appending: {} {} {} at {} {} {}".format(x, y, z, i, j, k))
                    annotations[(i, j, k)][z].append((x, y, z))
            
            nodules_coords = filtered_annotation['model_centers'].apply(ast.literal_eval)
            for nodule_coords in nodules_coords:
                for z, y, x in nodule_coords:
                    model_annot[(i, j, k)][z].append((x, y, z))


    
    # img = images[(0, 0)][1][:, :, 60].tolist()
    # print(np.min(img))
    # print(np.max(img))

    keys = list(images.keys())
    print("keys: ",keys)
    return render_template('result.html', keys=keys)


@app.route('/get_image')
def get_image():
    global images,annotations,model_annot


    i = int(request.args.get('i'))
    j = int(request.args.get('j'))
    k = int(request.args.get('k'))
    z = int(request.args.get('z'))
    print("index: ",i," ",j," ",k," ",z)
    
    img = images[(i, j)][k][z, :, :]

    org_anot = annotations[(i, j, k)][z] 
    pred_anot = model_annot[(i, j, k)][z] 

    mapped_img_data = np.clip((img + 30) * (255 / 305), 0, 255).astype(np.uint8).tolist()

    response_data = {
        "mapped_img_data": mapped_img_data,
        "org_anot": org_anot,
        "pred_anot": pred_anot
    }


    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)