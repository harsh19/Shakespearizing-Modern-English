##this will move all the model folder into a directory called models
#import 
import os
import shutil
def printCurretnDir():
    print(os.listdir("."))

def moveFileIntoDir(expression,directory):
    arr =os.listdir(".")
    for x in arr:
        if expression in  x:
            shutil.move(x,directory)
def moveTrainingCSV(expression,directory):
    moveFileIntoDir(".log",directory)
def removeDirectory(directory):
    the_dir = directory
    for x in the_dir:
        os.remove(x)
    os.remove(directory)
#moveFileIntoDir("epoch","./models")

removeDirectory("./models")

