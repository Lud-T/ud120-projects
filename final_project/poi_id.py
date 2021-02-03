#!/usr/bin/python

###Original Import 
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


###My import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pprint
from sklearn.linear_model import LinearRegression



pp = pprint.PrettyPrinter(depth=4)

'''First launch
(py2) PS C:\GitHub\ud120-projects\final_project> python poi_id.py
Traceback (most recent call last):
  File "poi_id.py", line 8, in <module>
    from tester import dump_classifier_and_data
  File "C:\GitHub\ud120-projects\final_project\tester.py", line 15, in <module>
    from sklearn.cross_validation import StratifiedShuffleSplit
ImportError: No module named cross_validation
Solution :
https://github.com/jkibele/OpticalRS/issues/7
'''

print "  "
print " -------"
print "| START |"
print " -------"
print "  "


'''
###My tests code
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
'''


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### First
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###Data exploration
print "  "
print " ------------------"
print "| Data Exploration |"
print " ------------------"
print "  "
print "People number : ", len(data_dict)
print "  "
print "features : ", len(data_dict.values()[0])
print "  "
print "features names: "
### Take the first name I get with 
### print data_dict.keys()[0]
pp.pprint(data_dict['METTS MARK'].keys())

print "  "
print "POIs : "
print "----"
count = 0
for employee in data_dict:
    if data_dict[employee]['poi'] == True:
        count += 1
        print employee
print "----"
print "Total POIs : ", count
print "----------"
print "  "
print "  "

###NaN check
print "Value/Nan check : "
print "---------------"
### create a dict with all the features to count the NaN
keysNaN = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())
for employee in data_dict:
        for key in data_dict[employee]:
            if data_dict[employee][key] == 'NaN':
                data_dict[employee][key] = np.nan
                #print type(data_dict[employee][key])
                keysNaN[key]+=1
print "NaN number for each feature:"          
pp.pprint(keysNaN)

###Zero check
print "---------------"
print "Value/Zero check : "
print "---------------"
### create a dict with all the features to count the NaN
keysZero = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())
for employee in data_dict:
        for key in data_dict[employee]:
            if data_dict[employee][key] == 0:
                keysZero[key]+=1
print "Zero number for each feature:"          
pp.pprint(keysZero)



'''
I check if the NaN count and the zero count can interfere If I replace the NaN by zero !!!
I think For every features without 0 I can put zero instead of NaN


Zero count
 'from_poi_to_this_person': 12,
 'from_this_person_to_poi': 20,

NaN count
 'from_poi_to_this_person': 60,
 'from_this_person_to_poi': 60,

'''

data_df = pd.DataFrame.from_dict(data_dict, orient='index')
pp.pprint(data_df.describe())
###At first, describe was not display the whole result I was expecting 
###I used my previous code to convert the NaN to zero, then it worked
###finally I tested the type of the value I was thinking it was NaN but it was string...I convert it to np.nan now it is float
###Now describe is working

'''At this stage I think I am already detecting outliers, one by one
I have done and understood the mini project on enron database which I suppose is relevant
but I feel it is too easy as we know the data well
What about, we know nothing about the data and their relations
'''
###Boxplot method to identify which feature may have outliers
###https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
###https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755

#sns.boxplot(data=data_df, orient="h")
#plt.show()

###Outliers everywhere !!!
###POI is not relevant
###Salary is
###Let's have a closer look to salary
#sns.boxplot(data_df["salary"])
#plt.show()

###I'll identify the highest

search = data_df["salary"] == data_df["salary"].max()
print data_df[search].index.values

###destroy the line TOTAL I found then check again for outliers
#data_dict.pop(data_df[search].index.values[0])
###check if it worked
#print "People number : ", len(data_dict)

data_df = data_df.drop(data_df[search].index)

pp.pprint(data_df.describe())

sns.boxplot(data_df["salary"])
plt.show()






#####################################################################################################






### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print "  "
print " -----"
print "| END |"
print " -----"
print "  "