import helper
import os
import io
from flask import Flask, render_template, request, session, flash, jsonify, redirect
from google.cloud import vision
import requests
import base64
import pickle
import joblib
from torchvision import transforms, models
import torch.nn.functional as F
import torch
from torch import nn
from PIL import Image
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # flask app root directory

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/clean', methods=['POST'])
def clean():
    text = request.form.get("memetext")
    clean_text = helper.clean_text(text)
    print('enter getJSONReuslt', flush=True)
    return render_template('cleantext.html', text=text, clean_text = clean_text)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    #receives an image from index.html and saves it to static/img/uploads
    #https://www.youtube.com/watch?v=6WruncSoCdI - https://www.youtube.com/watch?v=Y2fMCxLz6wM
    target = os.path.join(APP_ROOT, 'static')

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            destination = os.path.join(target, image.filename)
            image.save(destination)

            # Step1: Use Google API to extract text 
            # helper function get_text_from_image use google vision OCR to extract text from image
            text_extracted = helper.get_text_from_image(destination)

            # Step2: clean extracted text
            clean_text  = helper.clean_text(text_extracted)

            # Step3: load and use pre-trained bert tokenizer 
            tokenizer_path = os.path.join(APP_ROOT, 'static/models/bert_tokenizer.pkl')
            bert_tokenizer = pickle.load(open(tokenizer_path, 'rb'))
            sentence = bert_tokenizer(clean_text, return_tensors='pt', padding=True)
            input_ids = sentence['input_ids']
            print(type(input_ids))
            print(input_ids)
            attention_masks = sentence['attention_mask']

            
            # Step4: load and use pre-trained bert model to get text features
            bert_model_path = os.path.join(APP_ROOT, 'static/models/feature_extractor/text_features_model.pkl')
            bert_model = pickle.load(open(bert_model_path, 'rb'))
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_masks)
            text_features = outputs[1] #take the pooling layer output

            # Step5: define image transforms
            transforms_image = transforms.Compose([
                    transforms.Resize((224,224)), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
            
            # Step6: load and use pre-trained densenet model to get image features
            densenet_model = MyDenseNetConv()
            densenet_model.to('cpu')
            
            img = Image.open(destination)  #load image that the user uploaded
            img = transforms_image(img)
            img = np.asarray(img)
            img = torch.tensor(img).unsqueeze(0) # expand to 4-d to match input conditions
            image_features = densenet_model(img) # forward pass on the network to get images features from each batch

            # Step7: concat image embeddings + text embeddings
            concat = torch.cat((image_features[0], text_features[0]), dim=0)

            # Step8: create a dataframe and transpose it
            concat = pd.DataFrame(concat).astype("float").T

            # Step8: load xgb classifier (using same xgboost version - 0.9 - in google colab where it was exported and also here)
            xgboost_path = os.path.join(APP_ROOT, 'static/models/classifiers/concat_xgb_classifier.pkl')
            xgboost_model = pickle.load(open(xgboost_path, 'rb'))
            concat_prediction = xgboost_model.predict(concat)

            # Step9: text only classifier
            #bert_clf_path = os.path.join(APP_ROOT, 'static/models/classifiers/text_only_clf.pkl')
            #bert_clf = pickle.load(open(bert_clf_path, 'rb'))
            #outputs = bert_model(input_ids=input_ids, attention_mask=attention_masks)
            #text_features = outputs[1] #take the pooling layer output

    return  render_template("index.html", filename=image.filename, text_extracted=clean_text, concat_prediction=concat_prediction)
class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        image_model = models.densenet121(pretrained=True)

        #add a classifier of 2 classes to be able to match imported model architecture
        image_model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(256, 2))
        
        image_model.load_state_dict(torch.load('static/models/classifiers/image_only_clf.pt', map_location='cpu'))
        self.features = torch.nn.Sequential(*list(image_model.children())[:-1]) #1000
        
    def forward(self, x):
        # Since it outputs a 3d matrix [1024, 7, 7] we add one last avg pooling layer to get the 1024 final features
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x
'''
class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor = True):
        super(MyDenseNetConv,self).__init__()
        original_model = models.densenet121(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x
'''

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

