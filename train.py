import numpy as np
import features_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm


from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import sys
import joblib
import pandas as pd

labels=[]
features=[]
#file=open('Training Dataset.arff').read()
'''
list=file.split('\r\n')
#print (list)

data=np.array(list)

#print(data.shape)
data1=[i.split(',') for i in data]
#print (data1[0])
#data1=data1[0:-1]
#print(data1[0])
for i in data1:
	labels.append(i[30])

data1=np.array(data1)
print (data1.shape)
features=data1[:,:-1]
features=features[:,[0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,22,23,24,25,27,29]]
print (features)
features=np.array(features).astype(np.float)
'''
training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)
training_data = training_data[:,[0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,22,23,24,29,30]]
print ("Dataset shape")
print (training_data.shape)
#training_data = training_data[:,[0,1,2,3,4,30]]



#print (training_data.shape)
#print(training_data[0:1])

# Extract the inputs from the training data array (all columns but the last one)
inputs = training_data[:,:-1]
# Extract the outputs from the training data arra (last column)
outputs = training_data[:, -1]
training_inputs = inputs[:10000]
print ("Training Data shape "+str(training_inputs.shape)+"\n")
training_outputs = outputs[:10000]
#training_inputs = inputs
#training_outputs = outputs

testing_inputs = inputs[10000:]
print ("Testing data shape "+str(testing_inputs.shape)+"\n")

testing_outputs = outputs[10000:]
#print (training_inputs.shape)
#print (testing_outputs.shape)
#print("\n\n ""Random Forest Algorithm Results"" ")
clf4 = RandomForestClassifier(random_state=42)


clf4.fit(training_inputs, training_outputs)

predictions = clf4.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of  decision tree on testing data is: " + str(accuracy)+"\n")
'''
logreg = LogisticRegression()
logreg.fit(training_inputs, training_outputs)
predictions = logreg.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of your LR on testing data is: " + str(accuracy))




clf = svm.SVC(kernel='linear') 
clf.fit(training_inputs, training_outputs)
predictions = clf.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of your SVM on testing data is: " + str(accuracy))
'''
#url="https://www.youtube.com/"

url="https://www.google.com/"
print ("Input URL is : "+str(url)+"\n")
s=features_extraction.main(url)
print("Features returned\n")
print (s)
print("\n")
#s=[ 1 ,  1,  1,  1,  1, -1,  0, -1,  1, -1,  1,  0, -1, -1,  1,  1,  1, -1, -1,  1]

y=clf4.predict([s])
#print (y)
if y[0]==-1:
	print("Result :  Phishing \n")
else :
	print ("Result :  Not Phishing  \n")



#print(clf4.feature_importances_)
'''

#url="https://iqoption.com/land/start-trading/en/?aff=12545&afftrack=39354_&clickid=5bcb5038cf3e5c00011d2906"
#s=features_extraction.main(url)

s=[1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1]
print (s)
y=clf4.predict([s])
print (y[0])
prob=clf4.predict_proba([s])
print (prob)
'''
#joblib.dump(clf4, 'classifier/random_forest.pkl',compress=9)


#y=clf4.predict([[-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,1,0,-1,1,-1,1,1,1,1,1,1,-1,0,-1,1,1,1]])
#print (y[0])
'''
##### HAS TO BE CHANGED TO ALL ENTRIES OF THE DATASET
features_train=features
labels_train=labels
# features_test=features[10000:]
# labels_test=labels[10000:]



print("\n\n ""Random Forest Algorithm Results"" ")
clf4 = RandomForestClassifier(min_samples_split=7, verbose=True)
clf4.fit(features_train, labels_train)
importances = clf4.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf4.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(features_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# pred4=clf4.predict(features_test)
# print(classification_report(labels_test, pred4))
# print 'The accuracy is:', accuracy_score(labels_test, pred4)
# print metrics.confusion_matrix(labels_test, pred4)

#sys.setrecursionlimit(9999999)
joblib.dump(clf4, 'classifier/random_forest.pkl',compress=9)
'''
