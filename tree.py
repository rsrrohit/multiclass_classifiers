import numpy as np
import math
import time
import random
from queue import PriorityQueue


class Node(object):
    def __init__(self):
            self.value = "0"
            self.left = None
            self.right = None
            self.splitPoint = None

#Citation: The logic for following code was taken from
#https://paragmali.me/building-a-decision-tree-classifier/
def trainTree(trainData, level, trainDataMain):
    node = Node()
    if len(trainData) < 1:
        ran = random.randint(0, 3)
        mapping = {0 : "0", 1 : "90", 2 : "180", 3 : "270"}
        node.value = mapping[ran]
        return node #This node couldn't classify, so prey to god random returns the correct result.
    else:
        (isSame, classValue) = isSameClass(trainData) 
        if isSame or level < 0: # All elements are either classified in this node or max depth is reached.
            node.value = classValue # return the correct classification
            return node
    entropy, splitPoint = findTheSplit(trainData, trainDataMain)
    node.splitPoint = splitPoint
    [leftTrainData, rightTrainData] = splitTheData(trainData, splitPoint)
    node.left = trainTree(leftTrainData, level-1, trainDataMain)
    node.right = trainTree(rightTrainData, level-1, trainDataMain)
    return node
#Citation:  The logic for above code was taken from
#https://paragmali.me/building-a-decision-tree-classifier/

def splitTheData(trainData, splitPointData):
    splitPoint = splitPointData[0]
    row_index = splitPointData[1]#Dont need this right now.
    col_index = splitPointData[2]
    leftTrainData = []
    rightTrainData = []
    for row in range(len(trainData)):
        if trainData[row][col_index][0] < splitPoint:
            leftTrainData.append(trainData[row])
        else:
            rightTrainData.append(trainData[row])
    leftTrainData = np.array(leftTrainData)
    rightTrainData = np.array(rightTrainData)
    splittedData = [leftTrainData, rightTrainData]
    return splittedData

def findTheSplit(trainData, trainDataMain):
    featureQueue = PriorityQueue()
    trainDataTranspose = trainData.transpose()
    numberOfSamples = 20
    featureIndices = random.sample(range(2, 194), numberOfSamples)
    for i in featureIndices:
        row = trainDataTranspose[i]
        (entropy, split_point_data) = findTheRowSplit(row, trainData, trainDataMain)
        featureQueue.put((entropy, split_point_data))
    return featureQueue.get()

    #start_time = time.time()
    #print("Time elapsed in seconds = ", time.time() - start_time)

def findTheRowSplit(row, trainData, trainDataMain):
    N = len(trainData)
    queue = PriorityQueue()
    to_find = np.array([i for (i, j, k) in row])
    dummy, inds = np.unique(to_find, return_index=True)
    n = len(inds)
    for each_unique in inds:
        split_point = row[each_unique][0]
        left = []
        right = []
        for every_point in row:
            if every_point[0] < split_point:
                left.append(every_point)
            else:
                right.append(every_point)
        counts = [1, 1, 1, 1] #Counts for 0, 90, 180, 270
        entropy = 0  # To calculate the gini coefficient
        for lv in left:
            if trainDataMain[lv[1]][1] == "0":
                counts[0] += 1
            elif trainDataMain[lv[1]][1] == "90":
                counts[1] += 1
            elif trainDataMain[lv[1]][1] == "180":
                counts[2] += 1
            elif trainDataMain[lv[1]][1] == "270":
                counts[3] += 1

        n_left = len(left)
        n = sum(counts)
        entropy += n_left/N * ((counts[0]/n * counts[0]/n) + (counts[1]/n * counts[1]/n) + \
        (counts[2]/n * counts[2]/n) + (counts[3]/n * counts[3]/n))

        counts = [1, 1, 1, 1] #Counts for 0, 90, 180, 270
        for lv in right:
            if trainDataMain[lv[1]][1] == "0":
                counts[0] += 1
            elif trainDataMain[lv[1]][1] == "90":
                counts[1] += 1
            elif trainDataMain[lv[1]][1] == "180":
                counts[2] += 1
            elif trainDataMain[lv[1]][1] == "270":
                counts[3] += 1

        n_right = len(right)
        n = sum(counts)
        entropy += n_right/N * ((counts[0]/n * counts[0]/n) + (counts[1]/n * counts[1]/n) + \
        (counts[2]/n * counts[2]/n) + (counts[3]/n * counts[3]/n))

        queue.put((1-entropy, row[each_unique]))
    return queue.get()

def isSameClass(data):
    counts = [1, 1, 1, 1] #Counts for 0, 90, 180, 270. Initializing with 1 to avoid divide by zero exception
    for row in data:
        if row[1] == "0":
            counts[0] += 1
        elif row[1] == "90":
            counts[1] += 1
        elif row[1] == "180":
            counts[2] += 1
        elif row[1] == "270":
            counts[3] += 1

    maxi = max(counts)
    value = "0"
    if counts[0] == maxi:
        value = "0"
    elif counts[1] == maxi:
        value = "90"
    elif counts[2] == maxi:
        value = "180"
    elif counts[3] == maxi:
        value = "270"

    boo = False
    if(1.0 * maxi / sum(counts)) > 0.8: # If 80% of the items are classified then just declare the class which has a majority
        boo = True
    else:
        boo = False
    return (boo, value)

def train(fileName, model_file):
    trainDataMain = []
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
                trainDataMain.append(np.array(temp))
                rowNum += 1
    trainDataMain = np.array(trainDataMain)
    import pickle
    # with open('trainDataMain', 'wb') as fp:
    #     pickle.dump(trainDataMain, fp)

    # with open ('trainDataMain', 'rb') as fp:
    #     trainDataMain = pickle.load(fp)
    tree = trainTree(trainDataMain, 5, trainDataMain)
    with open(model_file, 'wb') as fp:  #Storing the model file
        pickle.dump(tree, fp)

def testThisRow(node, rowData):
    if node.left == None and node.right == None:
        return node.value
    else:
        [splitPoint, row_num, col_num] = node.splitPoint
        if rowData[col_num][0] < splitPoint:
            return testThisRow(node.left, rowData)
        else:
            return testThisRow(node.right, rowData)

def test(fileName, model_file):
    testDataMain = []
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
                testDataMain.append(np.array(temp))
                rowNum += 1
    testDataMain = np.array(testDataMain)
    import pickle
    with open (model_file, 'rb') as fp:
        tree = pickle.load(fp)
    correct = 0
    counter = 0
    with open("output.txt", "w") as fileFp:
        for testRow in testDataMain:
            prediction = testThisRow(tree, testRow)
            print("" + testDataMain[counter][0] + " " + prediction, file=fileFp)
            if(prediction == testRow[1]):
                correct += 1
            counter += 1
        fileFp.close()
    print("Samples tested = ", counter)
    print("Accuracy of decision tree = ", str((correct/counter)*100) + "%")
