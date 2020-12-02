import pandas as pd
import requests
import urllib.request,sys,time
from bs4 import BeautifulSoup
import re

f2=open("URLs.txt", "r")
text=[]

URL = "https://en.wikipedia.org/wiki/Analytic_number_theory"
print(URL)
try:
    page=requests.get(URL)
except Exception as e:
    # get the exception information
    error_type, error_obj, error_info = sys.exc_info()
    
    #print the link that cause the problem
    print ('ERROR FOR LINK:',URL)
    
    #print error info and line that threw the exception                          
    print (error_type, 'Line:', error_info.tb_lineno)

#time.sleep(2)
soup=BeautifulSoup(page.text,'html.parser')
for para in soup.find_all('p'):
    text.append(para.get_text())
text = ' '.join(text)
filename1="NEWS.csv"
f1=open(filename1,"w", encoding = 'utf-8')
f1.write(text)
print("")