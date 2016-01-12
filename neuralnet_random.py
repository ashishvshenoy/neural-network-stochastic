import random
import numpy as np
import math
from numpy import isreal
import time

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
	data = readArff("sonar.arff")
	N = neural_net()
	N.initializeNeuralNetObject(10, 0.1, 1000, data)
	samples = N.stratifiedSampler()
	i = 0
	testing_correct = 0
	training_correct = 0
	testing_wrong = 0
	training_wrong = 0
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
			if(classification>0.5):
				#print "Fold : "+str(i)+" Predicted: "+attributeList[-1].values[1] +" Actual: "+attributeList[-1].values[-1*d[-1]]+ " Confidence : "+str(classification)
				if(attributeList[-1].values[1]==attributeList[-1].values[-1*d[-1]]):
					testing_correct +=1
				else : 
					testing_wrong +=1
			else : 
				#print "Fold : "+str(i)+" Predicted: "+attributeList[-1].values[0] +" Actual: "+attributeList[-1].values[-1*d[-1]]+ " Confidence : "+str(classification)
				if(attributeList[-1].values[0]==attributeList[-1].values[-1*d[-1]]):
					testing_correct +=1
				else : 
					testing_wrong +=1
		for d in combinedSubSample:
			classification = N.computeOutputFromNetwork(d)
			if(classification>0.5):
				#print "Fold : "+str(i)+" Predicted: "+attributeList[-1].values[1] +" Actual: "+attributeList[-1].values[-1*d[-1]]+ " Confidence : "+str(classification)
				if(attributeList[-1].values[1]==attributeList[-1].values[-1*d[-1]]):
					training_correct +=1
				else : 
					training_wrong +=1
			else : 
				#print "Fold : "+str(i)+" Predicted: "+attributeList[-1].values[0] +" Actual: "+attributeList[-1].values[-1*d[-1]]+ " Confidence : "+str(classification)
				if(attributeList[-1].values[0]==attributeList[-1].values[-1*d[-1]]):
					training_correct +=1
				else : 
					training_wrong +=1
		#print correct
		i+=1
		N.setWeightsAndBiases(len(attributeList)-1)
		print N.counter
	avgTrainingAccuracy = (float(training_correct)/(training_correct+training_wrong))*100
	avgTestingAccuracy = (float(testing_correct)/(testing_correct+testing_wrong))*100
	print "Training Accuracy : "+str(avgTrainingAccuracy)
	print "Testing Accuracy : "+str(avgTestingAccuracy)

main()
		
		
			
		
	
