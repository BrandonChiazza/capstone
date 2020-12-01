# importing libraries
# !pip install google-cloud-vision
import os
from google.cloud import vision
import io
from PIL import Image
import urllib.request
import requests
import re

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def clean_text(text):
    #remove URL, html and user
    url = re.compile(r'http\S+')
    text = url.sub(r'', text)

    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(html, '', text)

    user = re.compile(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)')
    text = re.sub(user, '', text)

    return text

def get_text_from_image(image_location):
    """Receives a path to an image, and returns the text in it.
    - Uses Google Cloud Vision API"""

    # setting up the service account key as an environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.path.join(APP_ROOT, 'static/keys/windy-celerity-292623-b08a5b3c0c99.json')
    #print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    # CALL GOOGLE API TO GET THE TEXT FROM THE IMAGE
    client = vision.ImageAnnotatorClient()
    try:
        #read the image from static folder
        with open(image_location, "rb") as image_file:
            content = image_file.read()

        #format image for google vision
        img = vision.Image(content=content)

        #call api
        response = client.text_detection(image=img)
        text_extracted = response.text_annotations[0].description.replace('\n',' ')

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

    except requests.exceptions.RequestException as e:  
        raise SystemExit(e)
    

    # get the text from the response, and replace any line break with spaces
    return text_extracted
