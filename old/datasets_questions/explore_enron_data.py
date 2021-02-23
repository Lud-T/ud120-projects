#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint

pp = pprint.PrettyPrinter(depth=4)

#ValueError: insecure string pickle
#https://github.com/udacity/ud120-projects/issues/232
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print "Loaded"

#check people number
print "Dict length aka number of people : ",len(enron_data)

#check feature number
print "Dict length of the first dict aka number of features : ",len(enron_data.values()[0])

#Check features names 
print "Features of a person"
pp.pprint(enron_data.values()[0])

#Check people of interest
#Last results 'poi' True or False

print "Person of interest number"

count = 0
for user in enron_data:
    if enron_data[user]['poi'] == True:
        count += 1
print count



#Quizz How many POI with 
#../final_project/poi_names.txt
#../final_project/poi_email_addresses.py

print "Person of interest number with poi_names"

#https://python.sdv.univ-paris-diderot.fr/07_fichiers/
count = 0
#count if there is a y or n <= if the line is about a person or nothing or the label
with open("../final_project/poi_names.txt", "r") as filin:
    for ligne in filin:
        if ligne[1:2] == "y" :
            count += 1
        if ligne[1:2] == "n" :
            count += 1
print count

print 'James prentice total stock value'
print enron_data["PRENTICE JAMES"]['total_stock_value']

print 'How many email messages do we have from Wesley Colwell to persons of interest?'
print enron_data["COLWELL WESLEY"]['from_this_person_to_poi']


print 'What s the value of stock options exercised by Jeffrey K Skilling?'
print enron_data["SKILLING JEFFREY K"]['exercised_stock_options']