import numpy as np
import math
import pickle           #Serializes and deserializes objects 

#Essentially reads training data and stores it accordingly
def train(fileName, model_file):
    trainData = []
    trainDataFeatureMatrix = []
    with open(fileName, 'r') as fp:
        lines = fp.read().split('\n')
        rowNum = 0
        for line in lines:
            if len(line) > 1:
                elements = line.split(' ')
                temp = []
                temp.append(elements[0])
                temp.append(elements[1])
                featureList = [ [int(ttt), rowNum] for ttt in elements[2:]]
                columnNum = 2
                for eachFeature in featureList:
                    eachFeature.append(columnNum)
                    columnNum += 1
                temp = temp + featureList
                trainData.append(np.array(temp))
                trainDataFeatureMatrix.append(np.array([ int(ttt) for ttt in elements[2:]]))
                rowNum += 1
    trainData = np.array(trainData)
    trainDataFeatureMatrix = np.array(trainDataFeatureMatrix)
    model_file_data = [trainData, trainDataFeatureMatrix]
    with open(model_file, 'wb') as fp:
        pickle.dump(model_file_data, fp)       #Write the object obj to the open file object fp.                                             

def test(fileName, model_file):
    testData = []
    testDataFeatureMatrix = []
    with open(fileName, 'r') as fp:
        lines = fp.read().split('\n')
        rowNum = 0
        for line in lines:
            if len(line) > 1:
                elements = line.split(' ')
                temp = []
                temp.append(elements[0])
                temp.append(elements[1])
                featureRow = [ int(ttt) for ttt in elements[2:]]
                testDataFeatureMatrix.append(np.array(featureRow))
                temp = temp + featureRow
                testData.append(np.array(temp))
                rowNum += 1

    testData = np.array(testData)
    testDataFeatureMatrix = np.array(testDataFeatureMatrix)

    with open (model_file, 'rb') as fp:
        model_file_data = pickle.load(fp)
        [trainData, trainDataFeatureMatrix] = model_file_data
    correct = 0
    incorrect = 0
    counter = 0
    with open("output.txt", "w") as fileFp:
        for everyRow in testDataFeatureMatrix:
            prediction = knn(trainDataFeatureMatrix, everyRow, trainData)
            print("" + testData[counter][0] + " " + prediction, file=fileFp)
            if prediction == testData[counter][1]:
                correct += 1
            else:
                pass
                #print("Incorrectly classified " + testData[counter][0] + " " + prediction)
           
            counter += 1
        fileFp.close()
    print("k-nearest accuracy = ", format(correct/counter * 100, ".2f"))

def knn(data, test_row, trainData):
    distances = np.sqrt(np.sum(np.square(data-test_row), axis=1))
    indices = np.argsort(distances)
    counts = [1, 1, 1, 1] #Counts for 0, 90, 180, 270
    for i in range(50): # Here the value of k is 50
        if trainData[indices[i]][1] == "0":
            counts[0] += 1
        elif trainData[indices[i]][1] == "90":
            counts[1] += 1
        elif trainData[indices[i]][1] == "180":
            counts[2] += 1
        elif trainData[indices[i]][1] == "270":
            counts[3] += 1

    predicted_index = int(np.where(counts == np.amax(np.array(counts)))[0][0])
    mapping = {0 : "0", 1 : "90", 2 : "180", 3 : "270"}
    return mapping[predicted_index]
