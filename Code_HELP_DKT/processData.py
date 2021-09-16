import csv
import os
dirPath = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # ....../code

problemQmatrix = {}
personalizedQmatrix = {}
newLabel = {}


def ReadProblemQmatrix():
    global problemQmatrix
    with open(dirPath+'/Code_HELP_DKT/data/ModelInput'+'/problemQmatrix.CSV', 'r', encoding='utf-8-sig', newline='') as fileInput:
        reader = csv.reader(fileInput)
        for row in reader:
            problemQmatrix[row[0]] = row[1:]


def ReadPersonalizedQmatrix():
    global personalizedQmatrix
    with open(dirPath+'/Code_HELP_DKT/data/ModelInput'+'/P-matrix-out.CSV', 'r', encoding='utf-8-sig', newline='') as fileInput:
        reader = csv.reader(fileInput)
        for row in reader:
            personalizedQmatrix[row[0]] = row[1:]


def GetNewLabel():
    global problemQmatrix
    global personalizedQmatrix
    global newLabel
    for item in personalizedQmatrix.keys():
        problem = item.split('_')[1]
        problemConNum = 0
        for _ in problemQmatrix[problem]:
            problemConNum += int(_)
        personalizedConNum = 0
        for _ in personalizedQmatrix[item]:
            personalizedConNum += int(_)
        newLabel[item] = float(personalizedConNum/problemConNum)
    return


def WriteNewLabel():
    for path in ['train.CSV', 'test.CSV']:
        dataDir = {}
        with open(dirPath+'/Code_HELP_DKT/data/ModelInput/'+path, 'r', encoding='utf-8-sig', newline='') as fileInput:
            reader = csv.reader(fileInput)
            _ = 0
            for row in reader:
                dataDir[_] = row
                _ += 1
            i = 0
            while i < len(dataDir.keys()):
                count = 1
                tmp = []  # new label
                while (i+count) < len(dataDir.keys()) and (dataDir[i+count][0].split('_')[0] == 'c' or dataDir[i+count][0].split('_')[0] == 'b'):
                    tmp.append(newLabel[dataDir[i+count][0]])
                    count += 1
                if count != 1:
                    dataDir[i] = tmp
                i += count
        with open(dirPath+'/Code_HELP_DKT/data/ModelInput/'+path, 'w', encoding='utf-8-sig', newline='') as fileInput:
            writer = csv.writer(fileInput)
            for i in range(len(dataDir.keys())):
                writer.writerow(dataDir[i])
    return


def main():
    ReadProblemQmatrix()

    ReadPersonalizedQmatrix()

    GetNewLabel()

    WriteNewLabel()


if __name__ == "__main__":
    main()
