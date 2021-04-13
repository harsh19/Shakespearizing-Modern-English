##this will move all the model folder into a direcotry called models
#import 
import os
import shutil
def printCurretnDir():
    print(os.listdir("."))

def moveFileIntoDir(expression,direcotry):
    arr =os.listdir(".")
    for x in arr:
        if expression in  x:
            shutil.move(x,direcotry)

moveFileIntoDir("epoch","./models")
