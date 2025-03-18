import argparse
import csv
import sys

def splitData(data):
    header = []
    rows = []
    with open(data) as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader]

    training_data = rows[:int(len(rows) * 0.8)]
    testing_data = rows[int(len(rows) * 0.8):]

    with open("DATA/TrainingData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(training_data)

    with open("DATA/TestingData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(testing_data)

import random
def splitDataRandom(data):
    header = []
    rows = []
    with open(data) as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader]
    
    random.shuffle(rows)
    training_data = rows[:int(len(rows) * 0.8)]
    testing_data = rows[int(len(rows) * 0.8):]

    with open("DATA/TrainingData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(training_data)

    with open("DATA/TestingData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(testing_data)

def splitDataThree(data):
    header = []
    rows = []
    with open(data) as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader]

    training_data = rows[:int(len(rows) * 0.7)]
    validation_data = rows[int(len(rows) * 0.7):int(len(rows) * 0.85)]
    testing_data = rows[int(len(rows) * 0.85):]

    with open("DATA/TrainingData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(training_data)

    with open("DATA/ValidationData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(validation_data)

    with open("DATA/TestingData.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(testing_data)

def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input augument. Here are examples:")
    print("python " + sys.argv[0] + " --input ./DATA/CrabAgePrediction.csv")
    print("python " + sys.argv[0] + " --mode r --input DATA/CrabAgePrediction.csv")
    print("python " + sys.argv[0] + " --mode t --input DATA/CrabAgePrediction.csv")
    sys.exit(0)

def main():
    options = parser.parse_args()
    mode = options.mode
    print("mode is " + mode)
    if mode == '':
        """
        The training mode
        """
        inputFile = options.input
        if inputFile == '':
            showHelper()
        splitData(inputFile)
    elif mode == "r":
        inputFile = options.input
        if inputFile == '':
            showHelper()
        splitDataRandom(inputFile)
    elif mode == 't':
        inputFile = options.input
        if inputFile == '':
            showHelper()
        splitDataThree(inputFile)
    else:
        showHelper
    pass

if __name__ == "__main__":
    #------------------------arguments------------------------------#
    #Shows help to the users                                        #
    #---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
    default = '',    # default empty!
    help = 'Mode: empty for sequential split 80/20, r for random split 80/20, t for sequential split 70/15/15.')
    parser.add_argument('--input', dest='input',
    default = '',    # default empty!
    help = 'The input file to be split.')
    if len(sys.argv)<3:
        showHelper()
    main()