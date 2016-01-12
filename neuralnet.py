import random
import math
import time
import sys

class neural_net():
    weights = []
    data = []
    fold_info = {}
    index_info={}
    
    def __init__(self):
        self.epochs = 0
        self.learningRate = 0
        self.bias = 0.1
        self.counter=0
    
    def calculateSigmoid(self, z):
        if(z<-100):
            return 1
        try:
            return 1.0/(1.0+math.exp(-z))
        except:
            5
    
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
        
    def initializeNeuralNetObject(self, n,l,e,data):
        self.setWeightsAndBiases("")
        self.setData(self.convertClassificationToNumeric(data))
        self.setEpoch(e)
        self.setLearningRate(l)
        self.setFolds(n)
        
    def divideOnClassifications(self, dataSet):
        finalLists = list()
        list1 = []
        list2 = []
        if(len(dataSet)==0):
            return finalLists
        classification1 = dataSet[0][-1]
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
            if(self.fold>len(self.data)):
                self.fold = len(self.data)
            lengthOfEachSample = float(len(self.data))/self.fold
            proportion1 = (float(len(dividedList[0]))/(len(self.data)))*lengthOfEachSample
            proportion2 = 0
            dividedList.append([])
        else :
            lengthOfEachSample = float(len(self.data))/self.fold
            proportion1 = (float(len(dividedList[0]))/(len(self.data)))*lengthOfEachSample
            proportion2 = (float(len(dividedList[1]))/(len(self.data)))*lengthOfEachSample
        
        count = 0
        listOfSamples = [[]]
        while(count<self.fold):
            if(int(proportion1)==0 and int(proportion2)==0):
                break
            s = list()
            if(int(proportion1)!=0):
                s+=[dividedList[0].pop(random.randrange(len(dividedList[0]))) for _ in xrange(int(proportion1))]
            if(int(proportion2)!=0):
                s+=[dividedList[1].pop(random.randrange(len(dividedList[1]))) for _ in xrange(int(proportion2))]
            #s+=[dividedList[0].pop(random.randrange(len(dividedList[0]))) for _ in xrange(int(proportion1))]+[dividedList[1].pop(random.randrange(len(dividedList[1]))) for _ in xrange(int(proportion2))]
            it = 0
            if(len(s)==0):
                count+=1
                continue
            for d in s :
                self.fold_info[d[-1]] = count
                if(self.index_info.has_key(count)):
                    self.index_info[count].append(d[-1])
                else :
                    self.index_info[count] = [d[-1]]
                s[it].pop(-1)
                it+=1
            listOfSamples.append(s)
            count+=1
        iterator = 0
        if(len(listOfSamples)>1 and len(listOfSamples[0])==0):
            listOfSamples.pop(0)
        while(iterator<self.fold):
            if(len(dividedList[0])==0 and len(dividedList[1])==0):
                break
            if(len(dividedList[0])!=0):
                row = dividedList[0].pop()
                self.fold_info[row[-1]] = iterator
                if(self.index_info.has_key(iterator)):
                    self.index_info[iterator].append(row[-1])
                else : 
                    self.index_info[iterator] = [row[-1]]
                row.pop(-1)
                if(len(listOfSamples)-1<iterator):
                    listOfSamples.append([row])
                else : 
                    listOfSamples[iterator].append(row)
            if(len(dividedList[1])!=0):
                row = dividedList[1].pop()
                self.fold_info[row[-1]] = iterator
                if(self.index_info.has_key(iterator)):
                    self.index_info[iterator].append(row[-1])
                else : 
                    self.index_info[iterator] = [row[-1]]
                row.pop(-1)
                if(len(listOfSamples)-1<iterator):
                    listOfSamples.append([row])
                else : 
                    listOfSamples[iterator].append(row)
            iterator = (iterator+1)%self.fold
        if(len(listOfSamples[0])==0):
            listOfSamples.pop(0)
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
        test = self.learningRate*deltaE*-1
        self.bias +=(self.learningRate*deltaE*-1)
            
    
    def stochasticGradientDescent(self, dataSet):
        passes = 0
        output = 0
        while(passes<self.epochs):
            for d in dataSet:
                output = self.computeOutputFromNetwork(d)
                #error = self.calculateError(d[-1], output) #TODO : map the second classification as 1 and the first one as 0
                self.calculateGradientAndUpdateWeights(0, d, d[-1], output)
                output = 0
            passes+=1
        
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
    N = neural_net()
    trainingSet = sys.argv[1]
    no_of_folds = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    epochs = int(sys.argv[4])
    data = readArff(trainingSet)
    N.initializeNeuralNetObject(no_of_folds, learning_rate, epochs, data)
    samples = N.stratifiedSampler()
    i = 0
    prediction_list=[0]*len(data)
    while(i<len(samples)):
        if(i==0):
            if(len(samples)==1):
                subSample = [samples[0]]
            else : 
                subSample = samples[1:len(samples)]
        else:
            subSample = samples[0:i-1]+samples[i+1:len(samples)]
        combinedSubSample=list()
        for s in subSample :
            combinedSubSample+=s
        N.stochasticGradientDescent(combinedSubSample)
        p = 0
        for d in samples[i]:
            classification = N.computeOutputFromNetwork(d)
            prediction_list[N.index_info[i][p]]= classification
            p+=1
        p = 0
        i+=1
        N.setWeightsAndBiases(len(attributeList)-1)
    wrongCount = 0
    correctCount = 0
    z = 0
    for prediction in prediction_list :
        foldNo = int(N.fold_info[z])+1
        sno = z+1
        print sno ,
        print "Fold number: "+ str(foldNo),
        print " Predicted Class: ",
        if(prediction>0.5):
            print attributeList[-1].values[-1],
            predictedClass = attributeList[-1].values[-1]
        else : 
            print attributeList[-1].values[0],
            predictedClass = attributeList[-1].values[0]
        print " Actual Class: "+attributeList[-1].values[-1*data[z][-1]],
        if(predictedClass is attributeList[-1].values[-1*data[z][-1]]):
            correctCount+=1
        else :
            wrongCount+=1
        print " Confidence: "+str(prediction)
        z+=1
    accuracy = float(correctCount)/(correctCount+wrongCount)*100

main()
        
        
            
        
    
