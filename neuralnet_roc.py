import random
import numpy as np
import math
from operator import itemgetter

class neural_net():
    weights = []
    data = []
    fold_info = {}
    
    def __init__(self):
        self.epochs = 0
        self.learningRate = 0
        self.bias = 0.1
        self.counter=0
    
    def calculateSigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def setWeightsAndBiases(self, lengthOfInstance):
        self.weights= [0.1]*(len(attributeList)-1)
        self.bias = 0.1
    
    def convertClassificationToNumeric(self, dataSet):
        positiveClass = attributeList[-1].values[-1]
        negativeClass = attributeList[-1].values[0]
        index = 0
        while(index<len(dataSet)):
            if(dataSet[index][-1] == positiveClass):
                dataSet[index][-1] = 1
            else : 
                dataSet[index][-1] = 0
            index+=1
        return dataSet
    
    def setData(self, dataSet):
        self.data = dataSet
        
    def setEpoch(self, e):
        self.epochs = e
        
    def setLearningRate(self,l):
        self.learningRate = l
        
    def setFolds(self,n):
        self.fold = n
        
    def initializeNeuralNetObject(self, n,l,e,data,attributeList):
        self.setWeightsAndBiases("")
        self.setData(self.convertClassificationToNumeric(data))
        self.setEpoch(e)
        self.setLearningRate(l)
        self.setFolds(n)
        self.attributeList = attributeList
        
    def divideOnClassifications(self, dataSet):
        finalLists = list()
        list1 = []
        list2 = []
        if(len(dataSet)==0):
            return finalLists
        classification1 = 1
        classification2 = ""
        index_info = 0
        for d in dataSet :
            if(d[-1] == classification1):
                d.append(index_info)
                list1.append(d)
            else :
                d.append(index_info)
                list2.append(d)
            index_info+=1
        finalLists.append(list1)
        if(len(list2)==0):
            return finalLists
        else : 
            finalLists.append(list2)
            return finalLists
        
    def stratifiedSampler(self):
        dividedList = self.divideOnClassifications(self.data)
        if(len(dividedList) == 1):
            return self.data #TODO : Need to send 'fold' number of samples of the dataSet
        lengthOfEachSample = float(len(self.data))/self.fold
        
        proportion1 = (float(len(dividedList[0]))/(len(self.data)))*lengthOfEachSample
        proportion2 = (float(len(dividedList[1]))/(len(self.data)))*lengthOfEachSample
        
        
        count = 0
        listOfSamples = list()
        while(count<self.fold):
            s = list()
            s+=[dividedList[0].pop(random.randrange(len(dividedList[0]))) for _ in xrange(int(proportion1))]+[dividedList[1].pop(random.randrange(len(dividedList[1]))) for _ in xrange(int(proportion2))]
            it = 0
            for d in s :
                self.fold_info[d[-1]] = count
                s[it].pop(-1)
                it+=1
            listOfSamples.append(s)
            count+=1
        iterator = 0
        while(iterator<self.fold):
            if(len(dividedList[0])==0 and len(dividedList[1])==0):
                break
            if(len(dividedList[0])!=0):
                row = dividedList[0].pop()
                self.fold_info[row[-1]] = iterator
                row.pop(-1)
                listOfSamples[iterator].append(row)
            if(len(dividedList[1])!=0):
                row = dividedList[1].pop()
                self.fold_info[row[-1]] = iterator
                row.pop(-1)
                listOfSamples[iterator].append(row)
            iterator = (iterator+1)%self.fold
        return listOfSamples
    
    def computeOutputFromNetwork(self, instance): #TODO : Make sure the input is read as numeric in the list
        z = 0
        #z = np.dot(self.weights, output)+self.biases
        index = 0
        for w in self.weights :
            z+=instance[index]*w
            index+=1
        z+=self.bias
        output = self.calculateSigmoid(z)
        return output
    
    def calculateError(self, target_output, computed_output):
        return (float(math.pow((target_output-computed_output), 2)))/2
    
    def calculateGradientAndUpdateWeights(self, e, instanceVector, y, o):
        index = 0
        deltaE = -1*(y-o)*o*(1-o)
        for x_i in instanceVector:
            try:
                if("class" not in attributeList[index].Type):
                    #deltaE *=x_i
                    self.weights[index]+=self.learningRate*deltaE*x_i*-1
                    self.counter+=1
                else : 
                    break
                index+=1
            except :
                5
        self.bias +=(self.learningRate*deltaE*-1)
            
    
    def stochasticGradientDescent(self, dataSet):
        passes = 0
        output = 0
        while(passes<self.epochs):
            for d in dataSet:
                z = 0
                index = 0
                for w in self.weights :
                    z+=d[index]*w
                    index+=1
                z+=self.bias
                output = self.calculateSigmoid(z)
                self.calculateGradientAndUpdateWeights(0, d, d[-1], output)
                output = 0
            passes+=1
        5
        print self.weights
        
class Attribute:
    index = 0
    values = list()
    def __init__(self):
        self.setName("")
        self.setType("")
    def __str__(self):
        return str(self.Name)
    def setName(self, val):
        self.Name = val
    def setType(self, val):
        self.Type = val
    def setValues(self, val):
        self.values = val
    def getName(self):
        return str(self.Name)
    def getType(self):
        return str(self.Type)
    def getIndex(self):
        return self.index

attributeList=[]

def isfloat(value):
    try:
        v = float(value)
        return True
    except :
        return False

def readArff(filename):
    global attributeList
    arrfFile = open(filename)
    lines = [line.rstrip('\n') for line in arrfFile]
    data = [[]]
    index = 0
    for line in lines :
        if(line.startswith('@attribute')) :
            attributeLine = line
            attributeLineSplit = attributeLine.split(' ',2)
            if "{" not in attributeLineSplit[2] :
                attr = Attribute()
                attr.setName(attributeLineSplit[1].replace('\'',''))
                attr.setType("real")
                attr.index = index
                attributeList.append(attr)
            else : 
                attr = Attribute()
                attr.setName(attributeLineSplit[1].replace('\'',''))
                attr.setType("class")
                attr.index = index
                attributeValueList = attributeLineSplit[2].replace('{',"")
                attributeValueList = attributeValueList.replace('}',"")
                attributeValues = [x.strip(" ") for x in attributeValueList.split(",")]
                attr.setValues(attributeValues)
                attributeList.append(attr)
            index+=1
        elif(not line.startswith('@data') and not line.startswith('@relation')) :
            data.append([float(i) if isfloat(i) else i for i in line.split(',')])
    del data[0]
    return data

def main() :
    global attributeList
    data = readArff("sonar.arff")
    N = neural_net()
    N.initializeNeuralNetObject(10, 0.1, 100, data, attributeList)
    samples = N.stratifiedSampler()
    i = 0
    classificationSet = list()
    while(i<len(samples)):
        if(i==0):
            subSample = samples[1:len(samples)]
        else:
            subSample = samples[0:i-1]+samples[i+1:len(samples)]
        combinedSubSample=list()
        for s in subSample :
            combinedSubSample+=s
        
        random.shuffle(combinedSubSample)
        N.stochasticGradientDescent(combinedSubSample)
        for d in samples[i]:
            classification = N.computeOutputFromNetwork(d)
            classificationSet.append([classification, d[-1]])
        #print correct
        i+=1
        N.setWeightsAndBiases(len(attributeList)-1)
    sortedClassificationList = sorted(classificationSet, key=itemgetter(0))
    coordinateListTPR = list()
    coordinateListFPR = list()
    threshold = 0
    tp = 0
    fp = 0
    last_tp = 0
    m = 1
    num_pos = 0
    num_neg = 0
    for d in data : 
        if(attributeList[-1].values[-1*d[-1]]==attributeList[-1].values[-1]):
            num_pos+=1
        else : 
            num_neg+=1
    sortedClassificationList = list(reversed(sortedClassificationList))
    print sortedClassificationList
    if(sortedClassificationList[0][1]==1):
        tp+=1
    while(m<len(sortedClassificationList)):
        if(sortedClassificationList[m][0]!=sortedClassificationList[m-1][0] and sortedClassificationList[m][1]==0 and tp>last_tp):
            fpr = float(fp)/num_neg
            tpr = float(tp)/num_pos
            coordinateListTPR.append(tpr)
            coordinateListFPR.append(fpr)
            last_tp = tp
        if (sortedClassificationList[m][1] == 1):
            tp+=1
        else :
            fp+=1
        m+=1
    fpr = float(fp)/num_neg
    tpr = float(tp)/num_pos
    coordinateListFPR.append(fpr)
    coordinateListTPR.append(tpr)
    f = open("tpr_algo_list","w")
    f.write(str(coordinateListTPR))
    f = open("fpr_algo_list","w")
    f.write(str(coordinateListFPR))
    step = 0.1
    true_positive=0
    false_positive=0

    tpr_fpr_list=list()
    tpr_list = list()
    fpr_list = list()
    
    while(threshold<=1):
        for c in classificationSet:
            if(c[0]>=threshold):
                prediction = 1
                if(prediction==c[1]):
                    true_positive+=1
                else : 
                    false_positive+=1
            else : 
                prediction = 0
        tpr = float(true_positive)/num_pos
        fpr = float(false_positive)/num_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        tpr_fpr_list.append([tpr,fpr])
        true_positive=0
        false_positive=0
        threshold+=step
    print "****************************"
    file = open("tpr_fpr","w")
    file.write(unicode(str(tpr_fpr_list)))
    file = open("tpr_list", "w")
    file.write(unicode(str(tpr_list)))
    file = open("fpr_list","w")
    file.write(unicode(str(fpr_list)))

main()
        
        
            
        
    
