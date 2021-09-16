from functools import reduce
from typing import List

import torch
import torch.nn as nn
import argparse
import numpy as np
import random
from data import load_data
from HELP_DKT_Model import HELP_DKT_Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv
import os
import sys
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
random.seed(1)

dirPath = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # ....../code

parser = argparse.ArgumentParser(description='HELP-DKT model')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-linearWithQmatrix', type=str,
                    default='False',
                    help='whether the output linear layer with Qmatrix')
parser.add_argument('-multiLinearLayers', type=str,
                    default='True',
                    help='whether the num of output layers is more than 1 or not')
parser.add_argument('-one_hot', type=str,
                    default='True',
                    help='whether input layer combine vec and Qmatrix based on one-hot')
parser.add_argument('-inputConnectQmatrix', type=str,
                    default='False',
                    help='whether input vec conncet with Qmatrix')
parser.add_argument('-input_size', type=int,
                    default=20,
                    help='set input size for model and RunEpoch')
parser.add_argument('-inputQmatrixType', type=str,
                    default='P_Qmatrix',
                    help='input vec concrete or multiply with which Qmatrix type')
parser.add_argument('-inputMulQmatix', type=str,
                    default='True',
                    help='whether the input vec matmul with the Qmatrix')
parser.add_argument('-taskModel', type=str,
                    default='taskC',
                    help='run which task model')
parser.add_argument('-QmatrixType', type=str,
                    default='P_Qmatrix',
                    help='original dataSet,not simulate dataSet,use which Qmatrix type:personalized or original Qmatrix')
parser.add_argument('-masked', type=str,
                    default='True',
                    help='personalized or original Qmatrix')
parser.add_argument('-set2zero', type=str,
                    default='True',
                    help='whether set some of the output of the first linear layer to zero,\
                    which is not belong to current problem knowledge points')
parser.add_argument('-subQmatrix', type=str,
                    default='True',
                    help='whether the output of the first linear layer sub the Qmatrix')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-tryNum', type=int,
                    default=3,
                    help='the threshold for taskA, to decide whether the nest problems label is 1 or 0')
parser.add_argument('-epochs', type=int,
                    default=500,
                    help='Number of epochs to train')
parser.add_argument('-learning_rate', type=float,
                    default=0.05,
                    help='Learning rate')
parser.add_argument('-batch_size', type=int,
                    default=32,
                    help='Batch size for training')
parser.add_argument('-Qmatrix_size', type=int,
                    default=10,
                    help='the size of one Qmatrix')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-epsilon', type=float, default=0.1,
                    help='Epsilon value for Adam Optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.3,
                    help='Lambda for l2 loss')
parser.add_argument('-max_grad_norm', type=float, default=20,
                    help='Clip gradients to this norm')
parser.add_argument('-keep_prob', type=float, default=0.6,
                    help='Keep probability for dropout')
parser.add_argument('-hidden_layer_num', type=int, default=3,
                    help='The number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=200,
                    help='The number of hidden nodes')
parser.add_argument('-evaluation_interval', type=int, default=5,
                    help='Evalutaion and print result every x epochs')
parser.add_argument('-allow_soft_placement', type=bool,
                    default=True, help='Allow device soft device placement')
parser.add_argument('-log_device_placement', type=bool,
                    default=False, help='Log placement ofops on devices')
parser.add_argument('-output_size', type=int, default=6,
                    help='model linear layer output size')
parser.add_argument('-num_problems', type=list,
                    default=[10], help='num of problems in original data')
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
parser.add_argument('-train_data_path', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelInput/train.CSV',
                    help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelInput/test.CSV',
                    help='Path to the testing dataset')
parser.add_argument('-QmatrixPath', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelInput/P-matrix-out.CSV',
                    help='Path to the Q-matrix-out(created by getDataA/B.py')
parser.add_argument('-ProblemQmatrixPath', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelInput/problemQmatrix.CSV',
                    help='Path to the problemQmatrix(created by getDataA/B.py')
parser.add_argument('-model_output', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelOutput/result.txt',
                    help='Path to the testing dataset')
parser.add_argument('-id2problems', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelInput/ID2problem.CSV',
                    help='Path to the id2problems dataset')
parser.add_argument('-model_path', type=str,
                    default=dirPath+'/Code_HELP_DKT/data/ModelOutput/model.pth',
                    help='Path to the id2problems dataset')
args = parser.parse_args()

savedStdout = sys.stdout

Qmatrix = {}
problemQmatrix = {}
id2problems = {}


def RunPyFile():
    '''
    @description: Run getDataA.py/getDataB.py/getDataC.py
    @demands:
    @params:
    @return:
    '''
    if args.taskModel == 'taskA':
        os.system("python "+dirPath +
                  "/Code_HELP_DKT/getDataA.py -QmatrixType %s" % args.QmatrixType)
    elif args.taskModel == 'taskB':
        os.system("python "+dirPath +
                  "/Code_HELP_DKT/getDataB.py -QmatrixType %s" % args.QmatrixType)
    elif args.taskModel == 'taskC':
        os.system("python "+dirPath +
                  "/Code_HELP_DKT/getDataC.py -QmatrixType %s -Qmatrix_size %s" % (args.QmatrixType, args.Qmatrix_size))
    else:
        raise ValueError('args.taskModel ERROR!')
    os.system("python "+dirPath + "/Code_HELP_DKT/processData.py")

    return


def RepackageHidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(RepackageHidden(v) for v in h)


def RunEpoch(m, optimizer, students, batch_size, num_steps, num_skills, training=True, epoch=1):
    """Runs the model on the given data."""
    lr = args.learning_rate  # learning rate
    num_problems = args.num_problems
    total_loss = 0
    index = 0
    actual_labels = []
    pred_labels = []
    testSetVesMess = []  # save TEST set's message
    hidden = m.init_hidden(batch_size)
    count = 0
    batch_num = len(students) // batch_size
    rmse = 0
    auc = 0
    ability = []
    predsAll = []
    while (index + batch_size < len(students)):
        target_id: List[int] = []
        target_correctness = []
        target_id: List[int] = []
        target_correctness = []
        input_data = torch.FloatTensor(
            batch_size, num_steps, args.input_size).to(device)
        input_data = input_data.zero_()

        QmatrixInput = torch.FloatTensor(
            batch_size, num_steps, num_steps).to(device)
        QmatrixInput = QmatrixInput.zero_()

        problemQmatrixMask = torch.ByteTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixMask = problemQmatrixMask.zero_()

        problemQmatrixAbilityMask = torch.FloatTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixAbilityMask = problemQmatrixAbilityMask.zero_()

        problemQmatrixSub = torch.FloatTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixSub = problemQmatrixSub.zero_()

        problemQmatrixProd = torch.FloatTensor(
            batch_size, num_steps, args.Qmatrix_size
        ).to(device)
        problemQmatrixProd = problemQmatrixProd.zero_()

        vecMess = []
        for i in range(batch_size):
            student = students[index + i]
            problem_ids = student[1]
            correctness = student[2]

            if args.taskModel == 'taskA':
                for j in range(len(problem_ids)):
                    vec = []
                    vec, tmp = GetVec(correctness, student, j)
                    input_data[i, j, :] = torch.tensor(
                        vec, dtype=torch.float64).to(device)
                    vecMess.append(tmp)
                    testSetVesMess.append(tmp)

                    tmp = []  # input layer Qmatrix
                    for _ in Qmatrix[student[3+j][0]]:
                        if int(_) == 1:
                            tmp.append(int(_))
                        else:
                            tmp.append(random.uniform(-1e-10, 1e-10))
                    for _ in range(num_steps-args.Qmatrix_size):
                        tmp.append(random.uniform(-1e-10, 1e-10))
                    QmatrixInput[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64).to(device)

                    tmp = []  # masked
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(0)  # set 0
                        else:
                            tmp.append(1)
                    problemQmatrixMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    tmp = []  # problemQmatrixAbilityMask
                    for _ in Qmatrix[student[3+j][0]]:
                        if int(_) == 1:
                            tmp.append(1)  # torch.mul()
                        else:
                            tmp.append(0)
                    problemQmatrixAbilityMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

                    tmp = []  # sub
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(1)  # set 1 as original Qmatrix data
                        else:
                            tmp.append(0)
                    problemQmatrixSub[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    tmp = []  # sub
                    tmpInt = 0
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmpInt += 1
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(5/(1-float(tmpInt/args.Qmatrix_size)))
                        else:
                            tmp.append((1-float(tmpInt/args.Qmatrix_size)))
                    problemQmatrixProd[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

                tmp = GetNextProblem(
                    id2problems[student[3][0].split('_')[2]], problem_ids[0])
                target_correctness.append(tmp)
                actual_labels.append(tmp)
            elif args.taskModel == 'taskB' or args.taskModel == 'taskC':
                for j in range(len(problem_ids)-1):
                    vec, tmp = GetVec(correctness, student, j)
                    input_data[i, j, :] = torch.tensor(
                        vec, dtype=torch.float64).to(device)
                    vecMess.append(tmp)
                    testSetVesMess.append(tmp)

                    if args.taskModel == 'taskB' or args.taskModel == 'taskC':
                        target_id.append(i * num_steps +
                                         j + 0)
                    target_correctness.append(
                        float(correctness[j+1]))
                    actual_labels.append(int(float(correctness[j+1])))

                    tmp = []  # input layer Qmatrix
                    for _ in Qmatrix[student[3+j][0]]:
                        if int(_) == 1:
                            tmp.append(int(_))
                        else:
                            tmp.append(random.uniform(-1e-10, 1e-10))
                    for _ in range(num_steps-args.Qmatrix_size):
                        tmp.append(random.uniform(-1e-10, 1e-10))
                    QmatrixInput[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64).to(device)

                    tmp = []  # masked
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    problemQmatrixMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    tmp = []  # problemQmatrixAbilityMask
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            # ability & problemQmatrixAbilityMask
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    problemQmatrixAbilityMask[i, j, :] = torch.tensor(
                        tmp, dtype=torch.uint8
                    ).to(device)

                    tmp = []  # sub
                    tmpInt = 0
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmpInt += 1
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            # set 1 as original Qmatrix
                            tmp.append(float(tmpInt/args.Qmatrix_size))
                        else:
                            tmp.append(-0.5)
                    problemQmatrixSub[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

                    tmp = []  # sub
                    tmpInt = 0
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmpInt += 1
                    for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
                        if int(_) == 1:
                            tmp.append(5/(1-float(tmpInt/args.Qmatrix_size)))
                        else:
                            tmp.append(float(10))

                    problemQmatrixProd[i, j, :] = torch.tensor(
                        tmp, dtype=torch.float64
                    ).to(device)

        index += batch_size
        count += 1

        target_id2 = torch.tensor(
            target_id, dtype=torch.int64).to(device)
        target_correctness = torch.tensor(
            target_correctness, dtype=torch.float).to(device)

        if args.inputMulQmatix == 'True':
            input_data = torch.matmul(QmatrixInput.transpose(1, 2), input_data)

        if training:
            m.train()
            hidden = RepackageHidden(hidden)
            optimizer.zero_grad()
            output, hidden = m(input_data, hidden,
                               QmatrixInput, problemQmatrixMask, problemQmatrixSub, problemQmatrixAbilityMask, problemQmatrixProd, False)

            # Get predictions
            output = output.contiguous().view(-1)
            if args.taskModel == 'taskA':
                logitsPred = output
            elif args.taskModel == 'taskB' or args.taskModel == 'taskC':
                logitsPred = torch.gather(output, 0, target_id2).to(device)

            # preds
            if args.multiLinearLayers == 'False':
                preds = torch.sigmoid(logitsPred).to(device)
            else:
                preds = logitsPred.to(device)

            for p in preds:
                pred_labels.append(p.item())

            if args.multiLinearLayers == 'False':
                criterion = nn.BCEWithLogitsLoss().to(device)
            else:
                criterion = nn.BCELoss().to(device)
            loss = criterion(
                logitsPred, target_correctness)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(m.parameters(), args.max_grad_norm)

            optimizer.step()

            total_loss += loss.item()
        else:
            with torch.no_grad():
                m.eval()
                output, hidden, tmp = m(input_data, hidden,
                                        QmatrixInput, problemQmatrixMask, problemQmatrixSub, problemQmatrixAbilityMask, problemQmatrixProd, True)
                tmp_predsAll, tmp_ability = addVecMess(
                    output.tolist(), tmp, vecMess)
                predsAll += tmp_predsAll
                ability += tmp_ability

                output = output.contiguous().view(-1)
                if args.taskModel == 'taskA':
                    logitsPred = output
                elif args.taskModel == 'taskB' or args.taskModel == 'taskC':
                    logitsPred = torch.gather(output, 0, target_id2).to(device)

                # preds
                if args.multiLinearLayers == 'False':
                    preds = torch.sigmoid(logitsPred).to(device)
                else:
                    preds = logitsPred.to(device)
                for p in preds:
                    pred_labels.append(p.item())

                if args.multiLinearLayers == 'False':
                    criterion = nn.BCEWithLogitsLoss().to(device)
                else:
                    criterion = nn.BCELoss().to(device)
                loss = criterion(
                    logitsPred, target_correctness)
                total_loss += loss.item()

                hidden = RepackageHidden(hidden)

    rmse = sqrt(mean_squared_error(
        actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(
        actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    all_pred = []
    for _ in pred_labels:
        if _ >= 0.5:
            all_pred.append(1.0)
        else:
            all_pred.append(0.0)

    prec, rec, f1, _ = precision_recall_fscore_support(
        actual_labels, all_pred, average="binary")
    accuracy = metrics.accuracy_score(actual_labels, all_pred)

    if training == False and epoch % 100 == 99:
        writeAbilityPredsMess(ability, predsAll, epoch)

    return total_loss/(len(students)/batch_size), auc, accuracy, prec, rec, f1


def writeAbilityPredsMess(ability, predsAll, epoch):
    '''
    @description: Write ability messages
    @demands:
    @params:
    @return:
    '''
    if args.taskModel != 'taskC' or epoch < args.epochs - args.evaluation_interval:
        return

    resultParas = args.taskModel

    path = dirPath+'/Code_HELP_DKT/data/ModelOutput/studentAbilityMess-' + \
        str(resultParas)+'.CSV'
    with open(path, 'a+', encoding='utf-8-sig', newline='') as fileOutput:
        writer = csv.writer(fileOutput)
        writer.writerow(
            ['epoch:', epoch, 'taskModel:', args.taskModel, '\n\n'])
        writer.writerow(ability)


def addVecMess(output, ability, vecMess):
    if args.taskModel == 'taskA' or args.taskModel == 'taskB':
        return [], []
    result = []
    count = 0
    for i in range(len(ability)):
        result.append([])
        for j in range(len(ability[0])):
            result[i].append([])
            if any(ability[i][j]):
                result[i][j].append(vecMess[count])
                for _ in ability[i][j]:
                    result[i][j].append(_)
                count += 1

    preds = []
    count = 0
    for i in range(len(ability)):
        preds.append([])
        for j in range(len(ability[0])):
            preds[i].append([])
            if any(ability[i][j]):
                preds[i][j].append(vecMess[count])
                for _ in output[i][j]:
                    preds[i][j].append(_)
                count += 1

    return preds, result


def GetVec(correctness, student, j):
    '''
    @description: Get vec as model's input
    @demands: none
    @params:
    @return: vec
    '''
    tmp = []
    if args.inputQmatrixType == 'P_Qmatrix':
        for _ in Qmatrix[student[3+j][0]]:
            if int(_) == 1:
                tmp.append(int(_))
            else:
                tmp.append(random.uniform(-1e-10, 1e-10))
    else:
        for _ in problemQmatrix[student[3+j][0].split('_')[1]]:
            if int(_) == 1:
                tmp.append(int(_))
            else:
                tmp.append(random.uniform(-1e-10, 1e-10))

    vec = []
    if (int(float(correctness[j])) == 0):
        for _ in range(1, 11):
            vec.append(float(student[3 + j][_]))
        for _ in range(10):
            vec.append(random.uniform(-1e-10, 1e-10))
    else:
        for _ in range(10):
            vec.append(random.uniform(-1e-10, 1e-10))
        for _ in range(1, 11):
            vec.append(float(student[3 + j][_]))

    return vec, student[3+j][0]


def TakeEle(ele):
    return int(ele)


def GetNextProblem(problems, problemNum):
    '''
    @description: Get problem situation
    @demands: 
    @params: problems={}: The information of students
             problemNum: The current student's information
    @return: 1：tryNum of next problem <= args.tryNum
    '''
    tmp = []
    for i in problems.keys():
        tmp.append(i)
    tmp.sort(key=TakeEle)
    for i in tmp:
        if int(i) > int(problemNum):
            if int(problems[i]) <= args.tryNum:
                return 1
            else:
                return 0


def ReadQmatrix():
    '''
    @description: Get Qmatrix from args.Qmatrix
    @demands: args.Qmatrix is the path of file
    @params: none
    @return: none
    '''
    global Qmatrix
    with open(args.QmatrixPath, 'r', encoding='utf-8-sig') as fileInput:
        reader = csv.reader(fileInput)
        for i in reader:
            tmp = []
            for _ in i[1:]:
                tmp.append(_)
            Qmatrix[i[0]] = tmp
    with open(args.ProblemQmatrixPath, 'r', encoding='utf-8-sig') as fileInput:
        reader = csv.reader(fileInput)
        for i in reader:
            tmp = []
            for _ in i[1:]:
                tmp.append(_)
            problemQmatrix[i[0]] = tmp
    return


def ReadId2Problems():
    '''
    @description: Get information of Qmatrix-id
    @demands: none
    @params: none
    @return: none
    '''
    global id2problems
    with open(args.id2problems, "r", encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] not in id2problems.keys():
                id2problems[row[0]] = {row[1]: row[2]}
            else:
                id2problems[row[0]][row[1]] = row[2]
    return id2problems


def main():
    maxAuc = 0
    maxRmse = 0
    maxEpoch = 0
    maxAcc = 0
    maxPrec = 0
    maxRec = 0
    maxF1 = 0

    startTime = datetime.datetime.now()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    output_size = args.output_size
    train_students, train_max_num_problems, train_max_skill_num = load_data(
        train_data_path)
    num_steps = train_max_num_problems
    num_skills = train_max_skill_num
    num_layers = args.hidden_layer_num
    test_students, test_max_num_problems, test_max_skill_num = load_data(
        test_data_path)
    model = HELP_DKT_Model(
        'LSTM', args, num_skills, max(test_max_num_problems, train_max_num_problems)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=args.epsilon)

    ReadQmatrix()
    ReadId2Problems()

    # train & test model:
    aucList = ['auc']
    lossList = ['loss']
    for i in range(args.epochs):
        rmse, auc, accuracy, prec, rec, f1 = RunEpoch(
            model, optimizer, train_students, batch_size, max(test_max_num_problems, train_max_num_problems), num_skills, epoch=i)
        # Testing
        if ((i + 1) % args.evaluation_interval == 0):
            rmse, auc, accuracy, prec, rec, f1 = RunEpoch(model, optimizer, test_students,
                                                          batch_size, max(test_max_num_problems, train_max_num_problems), num_skills, training=False, epoch=i)
            print("TEST result:\n\tepoch:{}\n\tmodel:{}\n\tauc:{}\n\tacc:{}\n\tloss:{}\n\tprec:{}\n\trec:{}\n\tf1:{}\n\t".format(
                i+1, args.taskModel, auc, accuracy, rmse, prec, rec, f1))

            if auc > maxAuc:
                maxAuc = auc
                maxRmse = rmse
                maxEpoch = i+1
                maxAcc = accuracy
                maxPrec = prec
                maxRec = rec
                maxF1 = f1
            aucList.append(auc)
            lossList.append(rmse)

    # print AUC messages
    print("\n\nMAX TEST result:\n epoch:{}\nmodel:{}\nauc:{}\nacc:{}\nloss:{}\nprec:{}\nrec:{}\nf1:{}\n\n\n".format(
        maxEpoch, args.taskModel, maxAuc, maxAcc, maxRmse, maxPrec, maxRec, maxF1))
    torch.save(model.state_dict(), args.model_path)
    WriteResult([aucList, lossList], ['maxAuc', 'maxRmse', 'maxEpoch', 'maxAcc', 'maxPrec',
                                      'maxRec', 'maxF1'], [maxAuc, maxRmse, maxEpoch, maxAcc, maxPrec, maxRec, maxF1])


def WriteResult(modelResult, bestResultName, bestResult):
    # resultParas = args.taskModel + '-'

    with open(args.model_output, 'a+', encoding='utf-8-sig', newline='') as resultFile:
        # for tmpList in modelResult:
        #     tmp = ''
        #     for _ in tmpList[1:]:
        #         tmp = tmp + str(_) + ' '
        #     resultFile.write(resultParas+tmpList[0]+'\n')
        #     resultFile.write(tmp)
        #     resultFile.write('\n')
        tmp = "MODEL:" + args.taskModel+'\n'
        resultFile.write(tmp)
        for i in range(len(bestResultName)):
            tmp = ''
            tmp = '\t' + bestResultName[i] + '_' + str(bestResult[i])
            resultFile.write(tmp)
            resultFile.write('\n')


def PrintMessage():
    '''
    @description: Print messages
    @demands: none
    @params: none
    @return: none
    '''
    print('+-+-+-+-+-+-+-+-+-+')
    print('|run which model:', args.taskModel)
    print('|epoch:', args.epochs)
    print('|learning rate:', args.learning_rate)
    print('|batch size:', args.batch_size)
    print('+-+-+-+-+-+-+-+-+-+')


if __name__ == "__main__":

    PrintMessage()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('+-+-+-+-+-+-+-+-+-+')
        print("| RUNS ON CUDA!!! |")
        print('+-+-+-+-+-+-+-+-+-+')
    else:
        device = torch.device('cpu')
        print('+-+-+-+-+-+-+-+-+-+')
        print("| RUNS ON CPU!!!  |")
        print('+-+-+-+-+-+-+-+-+-+')

    RunPyFile()

    main()

    PrintMessage()
