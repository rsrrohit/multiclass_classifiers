from random import random, randint
import sys
import numpy as np
import math
import copy
import pickle

class ImageFiles:
    def __init__(self):
        pass

    testfile = {}
    trainfile = {}
    orientation = [0, 90, 180, 270]

#Referred resources
#https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python
# http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

def readfile(file_name):
    files = {}
    
    input_file = open(file_name, 'r')
    for line in input_file:
        data = line.split()
        img = np.empty(192, dtype=np.int)
        index = 2
        i = 0
        while i < 192:
            img[i] = int(data[index])
            index += 1
            i += 1
        files[data[0] + data[1]] = {"orient": int(data[1]), "img": img}
    input_file.close()
    return files
    #normalising

def normalise(files):
    all_files = []
    for i in files:
        img = [x * 1.0 / 255.0 for x in files[i]["img"].tolist()]
        orient = files[i]["orient"] / 90
        all_files.append(img + [orient])
    return all_files

def initialize_network(hidden_nodes):
    net = list()
    hidden_layer = [{'weights':[random() for i in range(193)]} for j in range(hidden_nodes)]
    net.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(hidden_nodes + 1)]} for j in range(4)]
    net.append(output_layer)
    return net

def activate(weights, inputs):
    output = 0
    for i in range(len(weights) - 1):
        output += weights[i] * inputs[i]
    output += weights[i]
    return output

def forwardpropogation(network, img):
    for layer in network:
        finaloutput = []
        for neuron in layer:
            x = activate(neuron['weights'], img)
            neuron['output'] = 1.0 / (1.0 + math.exp(-x))
            finaloutput.append(neuron['output'])
    return finaloutput

def backpropogation(network, expected):
    layer = network[1]
    errors = []
    for j in range(len(layer)):
        neuron = layer[j]
        # print("j = ", j)
        # print("len expected ", len(expected))
        # print("neuron['output'] ", neuron['output'])
        errors.append(expected[0] - neuron['output'])

    for j in range(len(layer)):
        neuron = layer[j]
        neuron['delta'] = errors[j] * neuron['output'] * (1 - neuron['output'])
    layer = network[0]
    errors = []
    for j in range(len(layer)):
        error = 0.0
        for neuron in network[1]:
            error += (neuron['weights'][j] * neuron['delta'])
        errors.append(error)
    for j in range(len(layer)):
        neuron = layer[j]
        neuron['delta'] = errors[j] * neuron['output'] * (1 - neuron['output'])

def updateweights(network, img, alpha):
    for neuron in network[0]:
        j = 0
        for j in range(192):
            neuron['weights'][j] += alpha * neuron['delta'] * img[j]
        neuron['weights'][j] += alpha * neuron['delta']

    inputs = [neuron['output'] for neuron in network[0]]

    for neuron in network[1]:
        j = 0
        for j in range(len(inputs) - 1):
            neuron['weights'][j] += alpha * neuron['delta'] * inputs[j]
        neuron['weights'][j] += alpha * neuron['delta']

def trainnnet(network, train, alpha, loops):
    for i in range(loops):
        totalerror= 0
        #print("train = ", train)
        for row in train:
            #print(row)
            #print("tpye row ", type(row))
            expected = [ row[-1] ]
            #expected[(i[-1])] = 1
            #totalerror += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            forwardpropogation(network, row)
            backpropogation(network, expected)
            updateweights(network, row, alpha)


def test_data(test, network):
    files_identified = 0

    results = []
    for img in test:
        outputs = forwardpropogation(network, img)
        result = outputs.index(max(outputs))
        results.append(result)
        if result == img[-1]:
            files_identified += 1
    #print(files_identified)
    #print(len(test))
    #accuracy = accuracy_metric(img[-1],results)
    #print(accuracy)
    print("Accuracy::" + str(files_identified * 100.0 / len(test) * 1.0))
    return results

alpha = 0.8
loops = 5
hidden_nodes = 5
img_file_names = []

def train(fileName, model_file):
    train_file = fileName #'train-data.txt'
    imf = ImageFiles()
    imf.trainfile = readfile(train_file)
    dataset = normalise(imf.trainfile)
    network = initialize_network(hidden_nodes)
    trainnnet(network, dataset, alpha, loops)
    

    with open(model_file, 'wb') as fp:  #Storing the model file
        pickle.dump(imf, fp)

def test(fileName, model_file):
    test_file = fileName # 'test-data.txt'
    with open (model_file, 'rb') as fp:
        imf = pickle.load(fp)
    imf.testfile = readfile(test_file)
    test_dataset = normalise(imf.testfile)
    network = initialize_network(hidden_nodes)
    results = test_data(test_dataset, network)
    with open ("output.txt","w") as f:
        for i in range(len(results)):
            print( str(results[i]), file=f)

    #print(img_file_names)
    # with open ("nnet_output.txt","w") as f:
    #     for i in range(len(img_file_names)):
    #         f.write(str(img_file_names[i]).split(".txt")[0] + ".txt"+ " " + str(results[i]))
#mode = "nnet"



# if mode == "nnet":
    
    

# def normalise(X,Y):
#     d1 = dict
#     Xn = [X * 1.0 / 255.0 for x in X]
#     Yn = Y / 90
#     d1 = {'Xn' : Xn, 'Yn':Yn}
#     return d1
