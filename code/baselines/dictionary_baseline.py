import sys

def readDict(dictFile):
    lines=dictFile.readlines()
    #lines=[x.lower() for x in lines]
    lines=[x.split() for x in lines]
    dictionary={}
    for line in lines:
        if len(line)==2:
            #print line[0]
            #print line[1]
            dictionary[line[1]]=line[0]
            #break

    return dictionary


if __name__=="__main__":
    dictFile=open(sys.argv[1])
    srcFile=open(sys.argv[2])
    trgFile=open(sys.argv[3],"w")
    
    dictionary=readDict(dictFile)

    srcLines=srcFile.readlines()
    srcLines=[x.lower() for x in srcLines]
    srcLines=[x.split() for x in srcLines]

    trgLines=[]
    for srcLine in srcLines:
        trgLine=[]
        for x in srcLine:
            nextWord=None
            if x in dictionary:
                nextWord=dictionary[x]
            else:
                nextWord=x
            trgLine.append(nextWord)
        trgLines.append(trgLine)

    print srcLines[2]
    print trgLines[2]

    trgLines=[" ".join(x) for x in trgLines]

    for trgLine in trgLines:
        trgFile.write(trgLine+"\n")

    trgFile.close()
