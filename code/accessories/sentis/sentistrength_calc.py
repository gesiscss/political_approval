#!/usr/bin/python


import subprocess
import shlex
from optparse import OptionParser
import gzip
import re
import os

file_path = os.path.dirname(os.path.abspath(__file__))
sentistrength_path = file_path + "/SentiStrength.jar"
sentistrength_data_path = file_path + "/data/"
#if options.filename == None:
#    raise Exception("Arquivo de entrada nescessario no parametro -f")


#Alec Larsen - University of the Witwatersrand, South Africa, 2012 import shlex, subprocess
def rate_sentiment(sentiString):
    if sentiString == "" or sentiString == None:
        return "0 0 0"
    #open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(shlex.split("java -jar " + sentistrength_path +" stdin sentidata " + sentistrength_data_path + " scale"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # p = subprocess.Popen(shlex.split("java -jar " + sentistrength_path +" stdin sentidata " + sentistrength_data_path + " trinary"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    # print("HIIII")
    # print(p)
    b = bytes(sentiString.replace(" ","+"), 'utf-8')
    #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1-5
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")
    stdout_text = stdout_text.rstrip().replace("\t"," ")
    stdout_text = stdout_text.replace('\r\n','')
    senti_score = stdout_text.split(' ')
    senti_score = list(map(float, senti_score))        
    senti_score = [int(i) for i in senti_score]
    senti_score = [1 if senti_score[i] >= abs(senti_score[i+1]) else -1 for i in range(0, len(senti_score), 3)]
    return senti_score[0]

    
if __name__ == "__main__":
    text = "you are the best"
    print(rate_sentiment(text))