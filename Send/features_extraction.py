#1 stands for legitimate

#-1 stands for phishing


import sys
import requests
from bs4 import BeautifulSoup
import urllib, bs4, re


#import urllib.request
# from selenium import webdriver
import urllib2, httplib
# import OpenSSL, sslbvnb
# import requests
from googlesearch import search

#from google import google
import whois
from datetime import datetime
import time

#import phishtank

import socket
proxyDict = { 
          'http'  : None, 
          'https' : None
        }


def having_ip_address(url):
    match=re.search('(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  #IPv4
                    '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  #IPv4 in hexadecimal
                    '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',url)     #Ipv6
    if match:
        #print match.group()
        return -1
    else:
        #print 'No matching pattern found'
        return 1

def url_length(url):
    if len(url)<54:
        return 1
    if len(url)>=54 and len(url)<=75:
        return 0
    else:
        return -1

def shortening_service(url):
    match=re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',url)
    if match:
        return -1
    else:
        return 1

def having_at_symbol(url):
    match=re.search('@',url)
    if match:
        return -1
    else:
        return 1

def double_slash_redirecting(url):
#since the position starts from, we have given 6 and not 7 which is according to the documen
    
    list=[x.start(0) for x in re.finditer('//', url)]

    #print ("do"+str(len(list)))
    if list[len(list)-1]>6:
        return -1
    else:
        return 1
    
    
def prefix_suffix(domain):
    match=re.search('-',domain)
    if match:
        return -1
    else:
        return 1

def having_sub_domain(url):
#Here, instead of greater than 1 we will take greater than 3 since the greater than 1 conition is when www and country domain dots are skipped
#Accordingly other dots will increase by 1
    if having_ip_address(url)==-1:
        match=re.search('(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5]))|(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',url)
        pos=match.end(0)
        url=url[pos:]
    list=[x.start(0) for x in re.finditer('\.',url)]
    if len(list)<=3:
        return 1
    elif len(list)==4:
        return 0
    else:
        return -1





def domain_registration_length(domain):
    expiration_date = domain.expiration_date
    #print (expiration_date)
    if expiration_date is None:
      return -1

    today = time.strftime('%Y-%m-%d')
    today = datetime.strptime(today, '%Y-%m-%d')
    #print (today)
    try:
        registration_length = abs((expiration_date - today).days)
    except:
        registration_length = abs((expiration_date[0] - today).days)
    
    #print ("re"+str(registration_length))
    #print (registration_length)
    if registration_length  <= 365:
        return -1
    else:
        return 1
    
    return 1
    
def favicon(wiki, soup, domain):
    for head in soup.find_all('head'):
        for head.link in soup.find_all('link', href=True):
            dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
            if wiki in head.link['href'] or len(dots) == 1 or domain in head.link['href']:
                return 1
            else:
                return -1
    return 1

def https_token(url):
    match=re.search('https://|http://',url)

    if match.start(0)==0:
        url=url[match.end(0):]
    
    match=re.search('http|https',url)
    if match:
        return -1
    else:
        return 1

def request_url(wiki, soup, domain):
   i = 0
   success = 0
   for img in soup.find_all('img', src= True):
      dots= [x.start(0) for x in re.finditer('\.', img['src'])]
      if wiki in img['src'] or domain in img['src'] or len(dots)==1:
         success = success + 1
      i=i+1

   for audio in soup.find_all('audio', src= True):
      dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
      if wiki in audio['src'] or domain in audio['src'] or len(dots)==1:
         success = success + 1
      i=i+1

   for embed in soup.find_all('embed', src= True):
      dots=[x.start(0) for x in re.finditer('\.',embed['src'])]
      if wiki in embed['src'] or domain in embed['src'] or len(dots)==1:
         success = success + 1
      i=i+1

   for iframe in soup.find_all('iframe', src= True):
      dots=[x.start(0) for x in re.finditer('\.',iframe['src'])]
      if wiki in iframe['src'] or domain in iframe['src'] or len(dots)==1:
         success = success + 1
      i=i+1
   success=i-success
   #print ("is"+str(success)+" "+str(i))
   try:
      percentage = success/float(i) * 100
   except:
       return 1
   #print(percentage)
   if percentage < 22.0 :
      return 1
   elif((percentage >= 22.0) and (percentage < 61.0)) :
      return 0
   else :
      return -1

def url_of_anchor(wiki, soup, domain):
    i = 0
    unsafe=0
    for a in soup.find_all('a', href=True):
    # 2nd condition was 'JavaScript ::void(0)' but we put JavaScript because the space between javascript and :: might not be
            # there in the actual a['href']
        
        if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (wiki in a['href'] or domain in a['href']):
            unsafe = unsafe + 1
        i = i + 1
        #print (a['href'])
    print (str(i)+" "+str(unsafe))
    try:
        percentage = (unsafe / float(i)) * 100
    except:
        return 1    
    #print (percentage)
    if percentage < 31.0:
        return 1
        # return percentage
    elif ((percentage >= 31.0) and (percentage < 98.0)):
        return 0
    else:
        return -1

# Links in <Script> and <Link> tags
def links_in_tags(wiki, soup, domain):
   i=0
   success =0
   for link in soup.find_all('link', href= True):
      dots=[x.start(0) for x in re.finditer('\.',link['href'])]
      if wiki in link['href'] or domain in link['href'] or len(dots)==1:
         success = success + 1
      i=i+1

   for script in soup.find_all('script', src= True):
      dots=[x.start(0) for x in re.finditer('\.',script['src'])]
      if wiki in script['src'] or domain in script['src'] or len(dots)==1 :
         success = success + 1
      i=i+1
   success=i-success
   try:
       percentage = success / float(i) * 100
   except:
       return 1
   #print(str(i)+" "+str(success)+" "+str(percentage))

   if percentage < 17.0 :
      return 1
   elif((percentage >= 17.0) and (percentage < 81.0)) :
      return 0
   else :
      return -1

# Server Form Handler (SFH)
###### Have written consitions directly from word file..as there are no sites to test ######
def sfh(wiki, soup, domain):
   for form in soup.find_all('form', action= True):
      if form['action'] =="" or form['action'] == "about:blank" :
         return -1
      elif wiki not in form['action'] and domain not in form['action']:
          return 0
      else:
            return 1
   return 1

#Mail Function
###### PHP mail() function is difficult to retreive, hence the following function is based on mailto ######
def submitting_to_email(soup):
   for form in soup.find_all('form', action= True):
      if "mailto:" in form['action'] :
         return -1
      else:
          return 1
   return 1

def abnormal_url(domain,url):
    #print (domain.domain_name)
    
    hostname=domain.domain_name[0].lower()
    #print(hostname)
    match=re.search(hostname,url)
    if match:
        return 1
    else:
        return -1
    


#IFrame Redirection
###### Checking remaining on some site######
def iframe(soup):
    for iframe in soup.find_all('iframe', width=True, height=True, frameBorder=True):
        if iframe['width']=="0" and iframe['height']=="0" and iframe['frameBorder']=="0":
            return -1
        
    return 1


def age_of_domain(domain):
    creation_date = domain.creation_date
    if creation_date is None:
      return -1

    expiration_date = domain.expiration_date
    if expiration_date is None:
      return -1
    try:
        ageofdomain = abs((expiration_date - creation_date).days)
    except:
        ageofdomain = abs((expiration_date[0] - creation_date[0]).days)

    #ageofdomain = abs((expiration_date - creation_date).days)
    #print (ageofdomain)
    if ageofdomain / 30 < 6:
        return -1
    else:
        return 1



def google_index(url):
    try:
      from googlesearch import search 
    except ImportError:
      print("No module named 'google' found") 
     
  
# to search 
    query = url
  
    for j in search(query, tld="com", num=1, stop=1, pause=2):
      return 1
    return -1
    


##### LINKS PONITING TO PAGE #####
def statistical_report(url,hostname):
    url_match=re.search('at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly',url)
    try:
        ip_address=socket.gethostbyname(hostname)
    except:
        print ('Connection problem. Please check your internet connection!')
##### 1st line is phishtank top 10 domain ips and 2nd, 3rd, 4th, 5th, 6th lines are top 50 domain ips from stopbadware #####
    ip_match=re.search('146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
                       '107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
                       '118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
                       '216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
                       '34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
                       '216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42',ip_address)
    if url_match:
        return -1
    elif ip_match:
        return -1
    else:
        return 1

def redirect(url):

  count=0
  httplib.HTTPConnection.debuglevel = 1
  '''
  while True:
    
    request = urllib2.Request(url)
    opener = urllib2.build_opener()
    f=opener.open(request)
    
    
    url = f.url
    count += 1
    if count>=2:
      break
    url = f.url
    count += 1
  '''
  f = requests.get(url, proxies=proxies)
  for resp in f.history:
    count+=1
  if count<=1:
    return 1
  elif count>=2 and count<4:
    return 0
  else:
    return -1 




def main(url):
    #url = sys.argv[1]
    #url="http://rtgfh.pro/azanhtzpoq/TE_Your_Health_Green_Coffee_IN_hi/?target=-7EBNQCgQAAANtIAMiCQAFAQEREQoRCQoRDUIRDRIAAX9hZGNvbWJvATE&al=2521&ap=17541&subid=VjN8MTQ4NDE3MTh8MTU0NjEyMXw4NDc4MTZ8MTU0MDEzMjgxOHw2MmEyYWU5NS03NDY0LTRmMWUtODEzNC0yYWFlZWRlYWQ0ZTJ8MTQuMTM5LjIzNi4yMTF8NHxzaD18YTA5ZTZjMWEzYTgwYzZlYjQyZGY4ZjJjMzIwOGRiYzM%3D&esub=-7EBRQCgQf83a5xgEDbSADIgkz4AiFRBPZCdUDfwcAAg_vj8xbEREKEQkiEQ1CEQ1aB2hrMgAAf2FkY29tYm__MjVjNDAwMGQAAzE1" 
    #url="http://www.hk.abchina.com/en/ebanking_4916/corporatebanking/201302/t20130204_316055.htm" 
    #url="https://tpbbayhyper.nl/torrent/9653996/Breaking_Bad_Season_1_Complete_720p.BRrip.Sujaidr_(pimprg)_"
    #print (url) 
    #print(redirect(url))
    page = requests.get(url, proxies=proxyDict)
    #page = requests.get(url)
    print ("Webpage request "+str(page.status_code))

    
    with open('markup.txt', 'r') as file:
        soup_string=file.read()

    soup = BeautifulSoup(page.content, 'html.parser')
    
    
    status=[]

    hostname = url
    h = [(x.start(0), x.end(0)) for x in re.finditer('https://|http://|www.|https://www.|http://www.', hostname)]
    z = int(len(h))
    if z != 0:
        y = h[0][1]
        hostname = hostname[y:]
        h = [(x.start(0), x.end(0)) for x in re.finditer('/', hostname)]
        z = int(len(h))
        if z != 0:
    
            hostname = hostname[:h[0][0]]
    
    
    #print (hostname)
    status.append(having_ip_address(url))
    print (url_length(url))
    status.append(url_length(url))
    status.append(shortening_service(url))
    status.append(having_at_symbol(url))
    status.append(double_slash_redirecting(url))
    
    status.append(prefix_suffix(hostname))
    status.append(having_sub_domain(url))
    #status.append(sslfinal_state(url))

    dns=1
    try:
        domain = whois.whois(hostname)
    except:
        dns=-1
    print ("dns is "+str(dns))
    #print (domain)
    if dns==-1:
        status.append(-1)
    else:
        status.append(domain_registration_length(domain))

    status.append(favicon(url,soup, hostname))
    status.append(https_token(url))
    status.append(request_url(url, soup, hostname))
    status.append(url_of_anchor(url, soup, hostname))
    status.append(links_in_tags(url,soup, hostname))
    status.append(sfh(url,soup, hostname))
    #print (submitting_to_email(soup))
    status.append(submitting_to_email(soup))
    
    
    if dns == -1:
        status.append(-1)
    else:
        status.append(abnormal_url(domain,url))
    
    # status.append(redirect(url))
    status.append(iframe(soup))
    #print (age_of_domain(domain))
    
    if dns == -1:
        status.append(-1)
    else:
        status.append(age_of_domain(domain))
    
    status.append(dns)
    
    #print (web_traffic(soup))
    #status.append(web_traffic(soup))
    #status.append(google_index(url))
    status.append(statistical_report(url,hostname))
    '''
    print ('\n1. Having IP address\n2. URL Length\n3. URL Shortening service\n4. Having @ symbol\n5. Having double slash\n' \
          '6. Having dash symbol(Prefix Suffix)\n7. Having multiple subdomains\n8. Domain Registration Length\n9. Favicon\n' \
          '10. HTTP or HTTPS token in domain name\n11. Request URL\n12. URL of Anchor\n13. Links in tags\n' \
          '14. SFH\n15. Submitting to email\n16. Abnormal URL\n(removed temporarily)11117. Redirect\n17. IFrame\n18. Age of Domain\n19. DNS Record\n20. Web Traffic\n' \
          '21. Google Index\n22. Statistical Report\n')
    
    '''
    #print (status)
   
    return status

if __name__ == "__main__":
    main()