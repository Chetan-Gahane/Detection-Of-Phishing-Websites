import joblib, features_extraction, sys

def main():
    #url=sys.argv[1]
    #url="http://rtgfh.pro/azanhtzpoq/TE_Your_Health_Green_Coffee_IN_hi/?target=-7EBNQCgQAAANtIAMiCQAFAQEREQoRCQoRDUIRDRIAAX9hZGNvbWJvATE&al=2521&ap=17541&subid=VjN8MTQ4NDE3MTh8MTU0NjEyMXw4NDc4MTZ8MTU0MDEzMjgxOHw2MmEyYWU5NS03NDY0LTRmMWUtODEzNC0yYWFlZWRlYWQ0ZTJ8MTQuMTM5LjIzNi4yMTF8NHxzaD18YTA5ZTZjMWEzYTgwYzZlYjQyZGY4ZjJjMzIwOGRiYzM%3D&esub=-7EBRQCgQf83a5xgEDbSADIgkz4AiFRBPZCdUDfwcAAg_vj8xbEREKEQkiEQ1CEQ1aB2hrMgAAf2FkY29tYm__MjVjNDAwMGQAAzE1"
    url="http://youtube.com"
    features_test=features_extraction.main(url)
    #feature_test=[1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1]

    clf = joblib.load('classifier/random_forest.pkl')

    pred=clf.predict([features_test])
    #prob=clf.predict_proba(features_test)
    #print (prob)
    # print 'Features=', features_test, 'The predicted probability is - ', prob, 'The predicted label is - ', pred
#    print "The probability of this site being a phishing website is ", features_test[0]*100, "%"


    if int(pred[0])==1:
        # print "The website is safe to browse"
        print ("SAFE")
    elif int(pred[0])==-1:
        # print "The website has phishing features. DO NOT VISIT!"
        print ("PHISHING")

    # print 'Error -', features_test

if __name__=="__main__":
    main()
