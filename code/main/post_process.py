# post processing step to replace unks with input word of highest attention
# CUDA_VISIBLE_DEVICES="" python post_process.py <checkpoint.test/valid> postProcess
# To get attention matrix for a specific instance in printed form (output element* input element)
# CUDA_VISIBLE_DEVICES="" python post_process.py <checkpoint.test/valid> write <exampleId>
# Important Note: This will create a file logs/attentionDump_S2S_<exampleId>.txt
# Important Note: Before running post processing, attention file should be available. This is generated during inference as alpha.p. Copy it to to <checkpoint.test>.alpha in /tmp

import sys
import pickle
import numpy as np

def padUp(line,finalLength,paddingMethod):
    words=line.split()
    words=["<s>",]+words+["</s>",]
    lineLength=len(words)
    padLength=finalLength-lineLength
    if padLength>0:
        if paddingMethod=="pre":
            words=["<p>"]*padLength+words
        elif paddingMethod=="post":
            words=words+["<p>",]*padLength
    elif padLength<0:
        words=words[:finalLength]

    return words


inputFile=open("../../data/test.modern.nltktok")
hypFile=open(sys.argv[1]+".output")
alpha=pickle.load(open(sys.argv[1]+".alpha","rb"))
mode=sys.argv[2]

if mode=="write":
    writeIndex=int(sys.argv[3])
elif mode=="postProcess":
    outFile=open(sys.argv[1]+".postProcessed","w")

inputLines=inputFile.readlines()
hypLines=hypFile.readlines()
inputLines=[padUp(line,25,"pre") for line in inputLines]
hypLines=[padUp(line,24,"post") for line in hypLines]

for i in range(len(inputLines)):
    if mode=="postProcess":
        hypLine=hypLines[i]
        inpLine=inputLines[i]
        attentionMatrix=alpha[i]
        inputStartIndex=-1
        for x in inpLine:
            if x!="<p>":
                break
            else:
                inputStartIndex+=1

        
        hypEndIndex=0
        for x in hypLine:
            if x=="<p>":
                break
            else:
                hypEndIndex+=1
        hypLine=hypLine[:hypEndIndex]
        inpLine=inpLine[inputStartIndex+1:]
        attentionMatrix=attentionMatrix[:hypEndIndex,inputStartIndex+1:]
        newHypLine=[]
        for k,x in enumerate(hypLine):
            if x=="<s>" or x=="</s>":
                continue
            elif x!="unk":
                newHypLine.append(x)
            else:
                attentionList=list(attentionMatrix[k])
                maxAttention=max(attentionList)
                maxAttentionIndices=[j for j,a in enumerate(attentionList) if a==maxAttention]
                maxAttentionIndex=maxAttentionIndices[0]
                xNew=inpLine[maxAttentionIndex]
                newHypLine.append(xNew)
        print newHypLine
        outFile.write(" ".join(newHypLine)+"\n")
        

    if mode=="write" and i==writeIndex:
        hypLine=hypLines[i]
        inpLine=inputLines[i]
        print hypLine
        print inpLine
        #attentionList=list((alpha[i])[9])
        #attentionList=[inputLines[i][k] if x>0.1 else "<b>" for k,x in enumerate(attentionList)]
        attentionMatrix=alpha[i]
        inputStartIndex=-1
        for x in inpLine:
            if x!="<p>":
                break
            else:
                inputStartIndex+=1
        print inpLine[inputStartIndex+1:]
        
        hypEndIndex=0
        for x in hypLine:
            if x=="<p>":
                break
            else:
                hypEndIndex+=1
        print hypLine[:hypEndIndex]
        print inpLine[inputStartIndex+1:]

        hypLine=hypLine[:hypEndIndex]
        inpLine=inpLine[inputStartIndex+1:]
        attentionMatrix=attentionMatrix[:hypEndIndex,inputStartIndex+1:]
       
        print attentionMatrix.shape
        print len(hypLine)
        print len(inpLine)
        
        attentionDump=open("logs/attentionDump_S2S_"+str(writeIndex)+".txt","w")
        for j in range(len(hypLine)):
            attentionList=list(attentionMatrix[j])
            print j,[inpLine[k] if x>0.1 else "<b>" for k,x in enumerate(attentionList)]
            #Renormalization
            #attentionSum=sum(attentionList)
            #attentionList=[x/attentionSum for x in attentionList]
            #Renorm end
            attentionListStr=" ".join([str(x) for x in attentionList])
            attentionDump.write(attentionListStr+"\n")
        
        pickle.dump(inpLine,open("logs/inpLine_S2S_"+str(writeIndex)+".p","wb"))
        pickle.dump(hypLine,open("logs/hypLine_S2S_"+str(writeIndex)+".p","wb"))

        attentionDump.close()

if mode=="postProcess":
    outFile.close()

