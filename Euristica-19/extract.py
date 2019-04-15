# information-extraction.py

import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
import os
import spacy
import sys
import csv
import json

os.chdir(r'C:\Users\gurvinder1.singh\Downloads\Data\HE_indore\resumes')

with open('techskill.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

string = open('CV of Binnu Thomas.txt').read()

#Function to extract names from the string using spacy
def extract_name(string):
    r1 = str(string)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(r1)
    for ent in doc:
        #print(ent.pos_)
        if(ent.pos_ == 'PROPN'):
            #print('name: ' + ent.text)
            return ent.text

def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names

if __name__ == '__main__':
    numbers = extract_phone_numbers(string)
    emails = extract_email_addresses(string)
    names = extract_names(string)
    name = extract_name(string)
    data = {}
    a=[]
    a.append(name)
    data['name']=a
    data['email']=emails
    data['phone']=numbers
    data['edu']=list()
    data['exp']=names
    #print('name: '+ name)
    #print('email: '+' , '.join(emails))
    #print('phone: '+' , '.join(numbers))
    #print('exp: '+ ','.join(names))
    #print('exp: '+list(set(names).intersection(set(your_list[0]))))
    #print(names)
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
