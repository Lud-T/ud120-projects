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
        count+=1
print count

