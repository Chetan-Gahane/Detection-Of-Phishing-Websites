import numpy as np
import features_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import sys
import joblib
import pandas as pd

training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

training_data = training_data[:,[0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,22,23,24,29,30]]
print ("Dataset shape")
print (training_data.shape)
#training_data = training_data[:,[0,1,2,3,4,30]]


#print (training_data.shape)
#print(training_data[0:1])

# Extract the inputs from the training data array (all columns but not the last one)
inputs = training_data[:,:-1]
# Extract the outputs from the training data array (only last column)
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

# Random Forest Classifier ...........................
clf4 = RandomForestClassifier(random_state=42)


clf4.fit(training_inputs, training_outputs)

predictions = clf4.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of  Decision tree on testing data is: " + str(accuracy)+"\n")

# Support Vector Machine ...............................................


clf = SVC(kernel='linear')
clf.fit(training_inputs, training_outputs)
predictions = clf.predict(testing_inputs)
accuracy1 = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of  Support vector Classifier on testing data is: " + str(accuracy1)+"\n")

# Naive - Bayes Classifier
gnb = GaussianNB()

gnb.fit(training_inputs, training_outputs)
predictions = gnb.predict(testing_inputs)
accuracy2 = 100.0 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of  Naive-Bayes Classifier on testing data is: " + str(accuracy2)+"\n")

#.......................................................
print("Accuracy of Random Forest Classifier is highest thats why we use Random Forest Classifier")

df=pd.read_csv("/home/manish/Desktop/verified_online.csv")
#url="https://www.youtube.com/"


#.....................................................................................
'''
print("Input data Sample "+str(df.shape)+"\n")
count=5
while(count<=10000) :
	url=df['url'][count]
	#man=df['submission_time'][count]
	#print(man)
	print ("Input URL is : "+str(url)+"\n")
	s=features_extraction.main(url)
	if s is None:
		count += 1
		continue
	print("Features returned\n")
	print (s)
	print("\n")
	#s=[ 1 ,  1,  1,  1,  1, -1,  0, -1,  1, -1,  1,  0, -1, -1,  1,  1,  1, -1, -1,  1]
	y=gnb.predict([s])
	count=count+1
	#print (y)
	if( y[0]==-1):
	    print("Result :  Legitimate (Not Phishing)\n")
	    print(".................................................................\n")
	else :
	    print ("Result : Phishing (Don't Visit)\n")
	    print(".................................................................\n")
'''
#.............................................................................

url=df['url'][4]
#url="https://www.facebook.com/"
#url="https://www.google.com/"
#url="https://viasimples.com.br/ead/ACESSO-DIGITAL-SEGURO/novidades-2019/home.php"
print ("Input URL is : "+str(url)+"\n")
s=features_extraction.main(url)
print("Features returned\n")
print (s)
print("\n")
#s=[ 1 ,  1,  1,  1,  1, -1,  0, -1,  1, -1,  1,  0, -1, -1,  1,  1,  1, -1, -1,  1]

y=clf4.predict([s])
#print (y)
if y[0]==-1:
	print("Result :  Phishing (Don't Visit)\n          It will be harmfull for you")
else :
	print ("Result : Legitimate (Not Phishing)  \n         Now You Can Visit this Website\n")


