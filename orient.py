###################################
# CS B551 Fall 2019, Assignment #4
#
# Code by: [Rohit Rokde - rrokde, Bhumika Agrawal - bagrawal, Aastha Hurkat - aahurkat]
#

import sys
import nearest
import tree
import nnet

if __name__ == "__main__":
    if(len(sys.argv) != 5):
        raise Exception("usage: orient.py test test_file.txt model_file.txt [model]")
    if sys.argv[1] == "train":
        if sys.argv[4] == "nearest":
            nearest.train(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == "tree":
            tree.train(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == "nnet":
            nnet.train(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == "best":
            nearest.train(sys.argv[2], sys.argv[3])
        else:
            print("Incorrect usage: Invalid model")

    elif sys.argv[1] == "test":
        if sys.argv[4] == "nearest":
            nearest.test(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == "tree":
            tree.test(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == "nnet":
            nnet.test(sys.argv[2], sys.argv[3])
        elif sys.argv[4] == "best":
            nearest.test(sys.argv[2], sys.argv[3])
        else:
            print("Incorrect usage: Invalid model")
    else:
        print("Incorrect usage: second parameter should be test or train")
