#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:32:01 2023

@author: vbr
"""
from retrieval import chat
from base64 import b64encode
from PIL import Image
import base64
import io
from PIL import Image
from io import BytesIO
import requests
from flask import Flask, request, send_file,jsonify
from flask_cors import CORS
import json

engine=chat()

app = Flask(__name__) # create the Flask app
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
count=0   

@app.route('/', methods=['POST'])
def dialog_turn():
    if request.is_json:
        data = request.json
        engine.dialog(data.get('utterance'),data.get('user_id'),data.get('session_id'),data.get('user_action'),data.get('interface_selected_product_id'),data.get('file'))
        response, response_imgs=engine.get_self_response()            
        if len(response_imgs)==0:
            responseDict = { "has_response": True, "recommendations":"",
                        "response":response, "system_action":""}
            jsonString = json.dumps(responseDict)
        else:
            responseDict = { "has_response": True, "recommendations":[{'image_path':img} for img in response_imgs],"response":response,
                            "system_action":""}
            jsonString = json.dumps(responseDict)
        return jsonString
        
# run app in debug mode on port 5000
if __name__ in '__main__':
    app.run(port=4000)
    