#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 22:29:06 2023

@author: vbr
"""
import pprint as pp
import requests
import json
from tqdm import tqdm
import pprint as pp
from opensearchpy import OpenSearch
from opensearchpy import helpers
from PIL import Image
import requests
import pandas as pd
import time
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from private import connection_credentials
from PIL import Image
from utils import embeddings_by_layer,attention_by_layer
import base64
import io
import numpy as np
import torch
import requests
from io import StringIO
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer,CLIPTextModel,CLIPVisionModel
import urllib.request
from model_load_inference import bert_intent_inference
from clip_explainability import explaianbility_retrieval

model_txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
model_vit = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")



### class retieval, we should get a message input, process it and return to 
# the user the best match from database.


class retrieval():
    def __init__(self,phase=1):
        # conenct with server
        host,port,index_name,user,password = connection_credentials()
        self.client = OpenSearch(
            hosts = [{'host': host, 'port': port}],
            http_compress = True,
            http_auth = (user, password),
            url_prefix = 'opensearch',
            use_ssl = True,
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False
        )
        self.phase=phase
        
        self.conversation={}
        
        self.user_id=None
        self.session_id=None
        
        
        # what user is looking for
        self.interface_selec=None
        self.file=None
        
        # chatbot answer
        self.response='hello'
      
    def check_img_input(self):
        '''here we will check if input is text,img or both'''
        # check for image files
        if self.file != None:
            self.file=self.file.split(',')[1]
            while len(self.file) % 4 != 0:
                self.file += "="
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(self.file, "utf-8"))))
            img.save('image.png')
            
            return True
        else:
            return False
        # check for text formating
        
        
    def set_query_denc(self):
        query_denc={'size': 3,
                  '_source': ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
          'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
          'product_short_description', 'product_attributes', 'product_image_path', 
          'product_highlights', 'outfits_ids', 'outfits_products','image_embedding','combined_embedding','text_embedding']}
        return query_denc    
        
    def create_embeddings(self):
        return 0
    
    def set_search_type(self,utterance,my_fields):
        utterance_list = utterance.split()
        query =[word for word in utterance_list if word != 'nan' ]
        query_txt=' '.join(word for word in query)
        fields =[field for field,word in zip(my_fields,utterance_list) if word != 'nan' ]
        
        if self.phase == 1:  
        # define search_type
            if (len(query) == 1) and (utterance[0] != '-') and ('nan' in utterance_list):
                search_type='word_based'
            elif (len(query) == len(my_fields) or (len(query) < len(my_fields) and ('nan' in utterance_list))) and  (utterance[0] != '-'):
                search_type = 'text_based'
            elif utterance[0] == '-':
                search_type = '_free_text'
                query_txt=query_txt[1:]
            else:
                search_type = 'free_text'
        
        # CONVERSATION STATE. save conversation iot get context when need it.
            if self.user_id in list(self.conversation.keys()):
                self.conversation[self.user_id].append(query_txt)
            else:
                self.conversation[self.user_id]=[query_txt]
                
            return search_type,fields,query
        
        # for the second phase project we will use image and text embedding for search
        elif self.phase == 2: # preprocessing for text input
            if self.check_img_input() == False:
                inputs = tokenizer([utterance], padding=True, return_tensors="pt")
                embeddings = F.normalize(model.get_text_features(**inputs))
                search_type='text_embedding'
                self.get_embedddings_analysis(inputs, search_type)
            else:
                if len(utterance) > 0: # preprocessing for combined features
                    inputs = tokenizer([utterance], padding=True, return_tensors="pt")
                    embeddings_txt=F.normalize(model.get_text_features(**inputs))
                    
                    qimg = Image.open("image.png")
                    input_img = processor(images=qimg, return_tensors="pt")
                    embeddings_img = F.normalize(model.get_image_features(**input_img))
                    
                    embeds = torch.tensor(embeddings_txt+embeddings_img)
                    embeddings = F.normalize(embeds, dim=0).to(torch.device('cpu'))   
                    search_type = 'combined_embedding'
                    
                else: # preprocessinf for image inou
                    print('here_we_have')
                    qimg = Image.open("image.png")
                    input_img = processor(images=qimg, return_tensors="pt")
                    embeddings = F.normalize(model.get_image_features(**input_img))
                    search_type = 'image_embedding'
                    self.get_patches_analysis(input_img,search_type)
            return search_type,embeddings

    
    def search(self,utterance,user_id,session_id,user_action,product_id_selec,file):
        query_denc = self.set_query_denc()
        
        # set user information and actions
        self.user_id=user_id
        self.interface_sele=product_id_selec
        self.file=file
        
        # check query in order to parse
        my_fields=['product_brand','product_gender','product_family','product_category','product_main_colour']
        if self.phase == 1:
            search_type,fields,query = self.set_search_type(utterance,my_fields)
            query_txt=self.conversation[self.user_id][-1]
        else:
            search_type,embeddings = self.set_search_type(utterance,my_fields)
        print(search_type)
        #  build the query/process based on argument search_type iot use open search  
        if search_type == 'word_based':
            query_denc['query'] = {'multi_match': {'query':query_txt,'fields': fields}}
                
        elif search_type == 'text_based':
            query_denc["query"] = {"bool": {"should": [ {"match": { f: q}} for f,q in zip(fields,query)]+[{ "match_all": {}}]}}

        elif search_type == '_free_text':
            query_denc['query'] = {'multi_match': {'query':query_txt,'fields': 'product_short_description'}}
                
        elif search_type == 'free_text':
            query_denc['query']={'multi_match': {'query':query_txt,'fields': fields}}
   
        else:
           query_denc["query"]={"knn": {search_type : {"vector": embeddings[0].detach().numpy(),"k": 5}}}
            
        # use open search to retrieve information base on tyoe query definied before
        response=self.client.search(body = query_denc,index = "farfetch_images")
        results = [r['_source']['product_image_path'] for r in response['hits']['hits']]   
        # set response ready to be prompted back to user client
        self.response= results
        
    def get_response(self):
        return self.response
    
    def get_embedddings_analysis(self,inputs,search_type):
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
        with torch.no_grad():
            if search_type == 'text_embedding' or search_type == 'image_embedding':
                outputs = model_txt(**inputs,output_attentions=True,output_hidden_states=True)
                # Cliptext model has 12 layers(default), for each of one we will get n tokens embeddings
                embeddings_by_layer(outputs['hidden_states'],tokens)
                #Cliptext model has 8 heads on attention mechanism (by default) for each layer
                attention_by_layer(outputs['attentions'], tokens, 4)
                
    def get_patches_analysis(self,inputs,search_type):
        with torch.no_grad():
            if search_type == 'image_embedding':
                outputs = model_vit(**inputs,output_attentions=True,output_hidden_states=True)
                # Cliptext model has 12 layers(default), for each of one we will get n tokens embeddings
                tokens =[f'patche_{i+1}' for i in range(outputs['attentions'][0].shape[2])]
                #Cliptext model has 8 heads on attention mechanism (by default) for each layer
                attention_by_layer(outputs['attentions'], tokens, 4)

                
    def explainability(self,given_image_path,utterance):
        explaianbility_retrieval(given_image_path, utterance)
        
    
    

class chat(): 
    def __init__(self):
        self.user_intent=None
        self.retrieval_base=retrieval(1)
        self.retrieval_emb=retrieval(2)
        self.utterance = ''
        self.answer = 'hi'
        self.response = []
        
        #history responses and answers
        self.answer_hist = []
        self.response_hist =[]

        # state conversation
        self.state = 'start'
        self.tracking_state = []
        
        
        self.graph={'start':{'user_neutral_greeting':('Hello! How can i help you?',0),
                             'user_request_get_products':("So you want a !\n Below are the 3 products,\n that most closely match the product\n you are looking for.",1),
                             'user_qa_product_description':('Sure! are u looking for which type of product',0),
                             'user_neutral_goodbye':('good bye, hope you can visit us again',0),'other':('sorry, as a Conversational Shopping Agent we are not able to satisfy your request. Are you interested in a ny product',0)},
                    'user_neutral_greeting':{'user_neutral_greeting':('Hello again! How can i help you?',0),
                             'user_request_get_products':("So you want a !\n Below are the 3 products,\n that most closely match the product\n you are looking for.",1),
                             'user_neutral_goodbye':('good bye, hope you can visit us again',0),
                             'explainability':('here is why the image was retrieved','explainability'),
                             'user_qa_product_description':('Sure! are u looking for which type of product',0),
                             'other':('sorry, as a Conversational Shopping Agent we are not able to satisfy your request. Are you interested in a ny product',0)},
                    'user_qa_product_description':{'user_neutral_goodbye':('good bye, hope you can visit us again',0),
                            'user_request_get_products':("So you want a !\n Below are the 3 products,\n that most closely match the product\n you are looking for.",1),
                            'user_neutral_goodbye':('good bye, hope you can visit us again',0),'explainability':('here is why the image was retrieved',2),
                            'other':('sorry, as a Conversational Shopping Agent we are not able to satisfy your request',0),
                            'user_qa_product_description':('Sure! are u looking for which type of product',0)},
                    'user_request_get_products':{'user_neutral_goodbye':('good bye, hope you can visit us again',0),
                             'user_request_get_products':("So you want a !\n Below are the 3 products,\n that most closely match the product\n you are looking for.",1),
                             'user_neutral_goodbye':('good bye, hope you can visit us again',0),'explainability':('here is why the image was retrieved',2),
                             'other':('sorry, as a Conversational Shopping Agent we are not able to satisfy your request. Are you interested in a ny product',0),
                             'user_qa_product_description':('Sure! are u looking for which type of product',0)},
                    'other':{'user_neutral_greeting':('Hello again! How can i help you?',0),
                             'user_request_get_products':("So you want a !\n Below are the 3 products,\n that most closely match the product\n you are looking for.",1),
                             'user_qa_product_description':('Sure! are u looking for which type of product',0),
                             'user_neutral_goodbye':('good bye, hope you can visit us again',0)}
                    }

    
    def dialog(self,utterance,user_id,session_id,user_action,interface_selected_product_id,file):
        self.response = []
        self.utterance=utterance
        self.set_intent()
        
        for k,v in self.graph[self.state].items():
            self.response = []
            if k == self.user_intent:
                if v[1] == 1:
                    self.retrieval_emb.search(self.tracking_state[-1][2],user_id,session_id,user_action,interface_selected_product_id,file)
                    self.response = self.retrieval_emb.get_response()
                    print('this',v[0][:13])
                    print('this1',self.tracking_state)
                    print('this',v[0][13:])
                    self.answer = v[0][:13] +" "+ self.tracking_state[-1][2].split()[0] + v[0][13:]
                elif v[1] == 2:
                    self.answer = v[0]
                    self.retrieval_emb.explainability(self.response[2],self.tracking_state[-2][2])
                
                else:
                    self.answer = v[0]
                    pass
                
                if k != 'user_neutral_goodbye':
                    self.state = k 
                else:
                    self.state = 'start'
                break
            
            else:
                pass
                
                
                
        
    def set_intent(self):
        intent,output_slots = bert_intent_inference(self.utterance)
        print(intent,output_slots)
        self.tracking_state.append([intent,"",""])
        if intent == 'user_request_get_products':
            print('isto',output_slots.items())
            for key,value in output_slots.items():

                self.tracking_state[-1][1] += key+" "
                self.tracking_state[-1][2] += value+" "
            self.user_intent = intent
        elif intent == 'user_qa_product_description':
            self.user_intent = intent
        elif intent == 'user_neutral_greeting':
            self.user_intent = intent
            
        elif intent == 'user_neutral_goodbye':
            self.user_intent = intent
            
        elif intent == 'user_inform_product_id':
            self.user_intent = intent
        
        else:
            if self.utterance == "why this product":
                self.user_intent = 'explainability'
            else:
                self.user_intent = 'other'
                
    def get_self_response(self):
        return self.answer,self.response