#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python standard library imports
import json
import csv
# All imports below are published on pypi.org
# regex 2021.11.10
import regex
# Pillow 9.0.0
from PIL import Image
# requests 2.27.1
import requests
# aiml 0.9.2
import aiml
# sklearn 0.0
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# pandas 1.3.5
import pandas as pd
# simpful 2.5.1
import simpful as sf
# nltk 3.6.7
from nltk.sem import Expression
from nltk.inference import ResolutionProver

# os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of tensorflow errors

# tensorflow
import tensorflow as tensor
from tensorflow import keras
# numpy
import numpy as np

# tkinter
import tkinter as tk
from tkinter import filedialog

# azure
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

print("Initialising chatbot...")

# Initialise knowledgebase (KB)
read_expr = Expression.fromstring
kb=[]
data = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

# Check KB integrity by checking if any contradictions
for i in range(len(data)):
    expr = read_expr('-' + str(kb[i]))
    answer=ResolutionProver().prove(expr, kb, verbose=False)
    if answer:
        print("A contradiction was spotted within the knowledgebase, exiting program...")
        quit()

# Create a kernel object
kern = aiml.Kernel()
kern.verbose(False)
kern.setTextEncoding(None)

# Define AIML file to use
kern.bootstrap(learnFiles="EuropeanCountriesChatBot.xml")

# Welcome user to chatbot
print("\nWelcome to the European Countries chat bot!")
print('''\nThis bot answers questions on countries within the continent Europe!
Extra functionality includes language fluency checker and capital/country/region storing.''')
print("\nStart by asking a question, or with a simple greeting! If stuck, try type 'tip'!")

# Define API URL's
# Free to use API developed by Alejandro Matos
cAPI_url = "https://restcountries.com/v3.1/name/"
aAPI_url = "https://restcountries.com/v3.1/alpha/"

# Define key and endpoint for cognitive services
cv_key = '33298418ecd44083b03fcfa983984744'
cv_endpoint = 'https://taskdcognitiveservice.cognitiveservices.azure.com/'

# Continent checker to check if countries continent is Europe
def checkEurope(country):
    succeeded = False
    response = requests.get(cAPI_url + country)
    if response.status_code == 200:
        response_json = json.loads(response.content)
        if response_json:
            sub = response_json[0]['region']
            if(sub == 'Europe'):
                succeeded = True
                return True
    if not succeeded:
        print("Sorry, this country does not exist, or is not in Europe!")
    return False

# Format kb inputs
def kbFormatting(object, subject):
    object = object.title()
    object = object.replace(" ", "")
    subject = subject.title()
    subject = subject.replace(" ", "")
    return object, subject

# Main loop of chatbot
while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    responseAgent = 'aiml'
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    if answer[0] == '#':
        params = answer[1:].split('$')

        # Split QA.csv into Q and A lists
        Qlist = []
        Alist = []

        with open('EuropeanQA.csv', newline='') as csvfile:
            QAreader = csv.reader(csvfile, delimiter='?', quotechar='|')
            for row in QAreader:
                Qlist.append(row[0])
                Alist.append(row[1][1:])

        cmd = int(params[0])
        if cmd == 0: # Chatbot exit
            print(params[1])
            break
        elif cmd == 2: # Currency of a country (using API)
            if checkEurope(params[1]):
                response = requests.get(cAPI_url + params[1])
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        curDICT = response_json[0]['currencies'] # grabs whole currency json and below lines grabs 'name' field
                        x = curDICT.keys()
                        m = regex.search(r"\'([A-Z]+)\'", str(x))
                        cur = response_json[0]['currencies'][m.group(1)]['name'] #['name'] = ['EUR']
                        print("The currency of", params[1].title(), "is:", cur)
            else:
                continue
        elif cmd == 3: # Borders of a country (using API)
            if checkEurope(params[1]):
                response = requests.get(cAPI_url + params[1])
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        try:
                            print("Searching country borders...")
                            bor = response_json[0]['borders'] # grabs country codes of all bordering countries
                        except:
                            print("No countries border that country!")
                            continue
                        borFN = []

                        i = 0
                        while i < len(bor): # searches country by country code and appends full name to list
                            response = requests.get(aAPI_url + bor[i])
                            if response.status_code == 200:
                                response_json = json.loads(response.content)
                                if response_json:
                                    name = response_json[0]['name']['common']
                                    borFN.append(name)
                            
                            i = i+1

                        seperator = ", "
                        
                        print("The countries that border", params[1].title(), "are:", seperator.join(borFN))
            else:
                continue
        elif cmd == 4: # Country code of country (using API)
            if checkEurope(params[1]):
                response = requests.get(cAPI_url + params[1])
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        cc = response_json[0]['cca3']
                        print("The country code of", params[1].title(), "is:", cc)
            else:
                continue
        elif cmd == 5: # Capital city of country (using API)
            if checkEurope(params[1]):
                response = requests.get(cAPI_url + params[1])
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        cap = response_json[0]['capital']
                        print("The capital of", params[1].title(), "is:", cap[0])
            else:
                continue
        elif cmd == 6: # Subregion of country (using API)
            if checkEurope(params[1]):
                response = requests.get(cAPI_url + params[1])
                if response.status_code == 200:
                    response_json = json.loads(response.content)
                    if response_json:
                        sub = response_json[0]['subregion']
                        print(params[1].title(), "is located in the subregion:", sub)
            else:
                continue
        elif cmd == 7: # Flag of country (using API)
            if checkEurope(params[1]):
                response = requests.get(cAPI_url + params[1])
                if response.status_code == 200:
                    print("The image of the flag will be displayed on your default image viewer...")
                    response_json = json.loads(response.content)
                    if response_json:
                        flag = response_json[0]['flags']['png']
                        img = Image.open(requests.get(flag, stream=True).raw)
                        img.show()
            else:
                continue
        elif cmd == 50: # I know that * is * + contradiction checker
            object,subject=params[1].split(' is ')
            object,subject = kbFormatting(object, subject)
            expr=read_expr(subject + '(' + object + ')')

            negexpr=read_expr('-' + subject + '(' + object + ')')
            neganswer=ResolutionProver().prove(negexpr, kb, verbose=False)

            if neganswer:
                print('Sorry, that contradicts with what I know!')
            else:
                kb.append(expr)
                print('OK, I will remember that',object,'is a', subject)
        elif cmd == 51: # Check that * is * + validity checker
            object,subject=params[1].split(' is ')
            object,subject = kbFormatting(object, subject)
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=False)
            if answer:
               print('Correct.')
            else:
               print('It may not be true. Let me check...')
               
               expr=read_expr('-' + subject + '(' + object + ')')
               answer=ResolutionProver().prove(expr, kb, verbose=False)

               if answer:
                   print('This is false')
               else:
                    print('I dont know')
        elif cmd == 52: # Fuzzy inference system on language fluency (code adapted from github -> aresio/simpful/examples/example_tip_mamdani.py)
            FS = sf.FuzzySystem(show_banner=False)

            rules = []
            with open("fuzzyRules.txt", "r") as file:
                for line in file:
                    rules.append(line.strip())
            
            FS.add_rules(rules)

            # Define fuzzy sets + output fuzzy set and linguistic variables
            S_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="low")
            S_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=5, c=10), term="average")
            S_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="high")
            FS.add_linguistic_variable("Speaking", sf.LinguisticVariable([S_1, S_2, S_3], concept="Speaking fluency", universe_of_discourse=[0,10]))

            F_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="low")
            F_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=5, c=10), term="average")
            F_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="high")
            FS.add_linguistic_variable("Writing", sf.LinguisticVariable([F_1, F_2, F_3], concept="Writing fluency", universe_of_discourse=[0,10]))

            T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=10), term="low")
            T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=10, c=20), term="average")
            T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=10, b=20, c=25, d=25), term="high")
            FS.add_linguistic_variable("Fluency", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,25]))

            language = input('What language would you like to know your fluency rating on? ')

            while True: # Validation checks to ensure number is within range and an integer
                try:
                    speaking = int(input('From 0-10, how fluent are you in speaking ' + language + ': '))
                    writing = int(input('From 0-10, how fluent are you in writing ' + language + ': '))
                except ValueError:
                    print("Inputs must be integers!")
                    continue
                else:
                    if (speaking < 0 or speaking > 10 or writing < 0 or writing > 10):
                        print("One of the inputs are not within the range required, 0-10.")
                        continue
                    else:
                        break

            # Set fuzzy variables
            FS.set_variable("Speaking", speaking)
            FS.set_variable("Writing", writing)

            # Perform mamdani inference
            result = FS.Mamdani_inference(["Fluency"])
            fluency = result.get('Fluency')

            # Output sentence based on fluency result
            if fluency >= 0 and fluency < 8:
                print('You do not know ' + language + ' fluently.')
            elif fluency >= 8 and fluency < 11:
                print('You are entry level at ' + language + '. However, you are not fluent.')
            elif fluency >= 11 and fluency < 14:
                print('You are intermediate level at ' + language + '. However, you are not fluent.')
            elif fluency >= 14 and fluency < 16:
                print('You are advanced level at ' + language + '. However, you are not fluent.')
            else:
                print('You are fluent in ' + language)
        elif cmd == 53: # CNN model used to identify what world wonder chosen image is
            # Select an image from file explorer
            try:
                root = tk.Tk()
                root.filename = filedialog.askopenfilename(initialdir="test_data/", title="Select An Image", filetypes=(("JPG files", "*.jpg"),("All Files", "*.*")))
            except (KeyboardInterrupt, EOFError) as e:
                print("No image selected.")
                continue
            root.destroy()
            imgPath = root.filename

            # Prompt user with question over which option they would like to do
            decision = input("Would you like to analyse or identify this image?\n> ")

            if decision.lower() == "analyse":
                # Client from computer vision service
                cv_client = ComputerVisionClient(cv_endpoint, CognitiveServicesCredentials(cv_key))

                # Description from computer vision service
                imgStream = open(imgPath, "rb")
                description = cv_client.describe_image_in_stream(imgStream)

                for caption in description.captions:
                    print("Azure Cognitive Web Service.") 
                    print("This image is most likely {}. This is said with {:.2f} percent confidence.".format(caption.text, caption.confidence*100))

            elif decision.lower() == "identify":
                # Load WorldWodnerModel.h5 model
                model= tensor.keras.models.load_model("WorldWonderModel.h5")

                # Load selected image into correct format for model
                img = tensor.keras.utils.load_img(imgPath, target_size = (180,180))
                imgArray = tensor.keras.utils.img_to_array(img)
                imgArray = tensor.expand_dims(imgArray, axis = 0)

                # Use model to predict score
                output = model.predict(imgArray)
                score = tensor.nn.softmax(output[0])

                # Output predicted class with confidence percentage
                class_names = ['The Christ The Redeemer Statue', 'The Eiffel Tower', 'The Pyramids of Giza', 'Stonehenge']
                print("This image is most likely {}. This is said with {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))

                # if output[0][0] == 1:
                #     prediction = "Christ the Redeemer"
                # elif output[0][1] == 1:
                #     prediction = "The Eiffel Tower"
                # elif output[0][2] == 1:
                #     prediction = "Pyramid of Giza"
                # elif output[0][3] == 1:
                #     prediction = "Stonehenge"
                # else:
                #     prediction = "Unknown"
                # print("This image is most likely the world wonder: " + prediction)

        elif cmd == 99:
            # Similarity conversation using BoW, tf-idf, and cosine similarity
            userInputList = [userInput]
            tf = TfidfVectorizer(sublinear_tf=True, stop_words='english') # Sublinear applies sublinear tf scaling i.e. replace tf with 1+log(tf) // stop words implemented
            tf_qlist = tf.fit_transform(Qlist) # TfidfVectorizer performs bag of words and tf-idf on questions from EuropeanQA.csv
            tf_input = tf.transform(userInputList) #TfidfVectorizer performs bag of words and tf-idf on user input
            cosineSimilarities = cosine_similarity(tf_qlist, tf_input).flatten() # Perform cosine similarity based on tf-idf input and tf-idf questions from EuropeanQA.csv
            bestMatch = cosineSimilarities.argsort()[-1] # Sort smallest value to largest value and grab last value in list
            if (cosineSimilarities[bestMatch] > 0.5): # Only output if cosine similarity score is above 50%
                ans = [Alist[int(bestMatch)]]
                print("This may help:", ans[0])
            else:
                print("Sorry I could not understand that, please try again.")
    else:
        print(answer)