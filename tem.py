import sys
import newtrain

def main():
	url=sys.argv[1]
	#print (url)
	x=newtrain.main(url)
	print (str(x))


if __name__=="__main__":
    main()