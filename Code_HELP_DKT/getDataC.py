import csv
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-QmatrixType', type=str, default='P_Qmatrix')
parser.add_argument('-Qmatrix_size', type=int, default=10)
args = parser.parse_args()

dirPath = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # ....../code

filename = dirPath+'/Data/Program_Vector_Embeddings.CSV'
file_train_out = dirPath+'/Code_HELP_DKT/data/ModelInput/train.CSV'
file_test_out = dirPath+'/Code_HELP_DKT/data/ModelInput/test.CSV'


def get_num(index, rows):
    '''
    description: Get the current student's total submmisions
    demands: 
    params: index: The index of dataset
            rows: Dataset
    return: The total number(int)
    '''
    count = 0
    id = rows[index][0]
    while index < len(rows) and rows[index][0] == id:
        count += 1
        index += 1
    return count


def takeELE2(input):
    return int(input[1].split('_')[-1].split('.')[0])


def WriteProblemGroup(input, index, type, tmpNum, writer):
    '''
    @description: Group problems
    @demands:
    @params: types: The current student's problem
             tmpNum: The total number of the current student's submissions
    @return: none
    '''
    rows = []
    for i in range(tmpNum):
        rows.append(input[index+i])
    rows.sort(key=takeELE2)
    num = 0
    count = 0
    vecs = []
    correctness = []
    types = []
    index = 0
    while index < len(rows) and count < tmpNum:
        if rows[index][0] == '':
            index += 1
            continue
        if rows[index][2] == type:
            num += 1
            if rows[index][1][0] == 'c':
                correctness.append(1)
            else:
                correctness.append(0)
            types.append(rows[index][2])
            vecs.append(rows[index][1])
            for tmp in rows[index][4].strip().split(' '):
                vecs.append(tmp)
            count += 1
            index += 1
        else:
            index += 1
            count += 1

    return num, types, correctness, vecs


def write_group(rows, index, writer):
    '''
    @description: Group dataset based on problems
    @param: 
    @return: The index of dataset
    '''
    tmp_num = get_num(index, rows)
    types = []
    for _ in range(tmp_num):
        if rows[_+index][2] not in types:
            types.append(rows[_+index][2])
    f_num = 0
    f_types = []
    f_correctness = []
    f_vecs = []
    for _ in types:
        num_, types_, correctness_, vecs_ = WriteProblemGroup(
            rows, index, _, tmp_num, writer)
        f_num += num_
        f_types += types_
        f_correctness += correctness_
        f_vecs += vecs_

    writer.writerow([f_num])
    writer.writerow(f_types)
    writer.writerow(f_correctness)

    for i in range(f_num):
        writer.writerow(f_vecs[i * 11:(i + 1) * 11])

    index = index+tmp_num

    return index


def takeEle(elem):
    return elem[0]


type_data = {}
problemNum = {}
problemNumConv = {}


def getGroup(input):
    '''
    @description: Get dataset
    @demands: 
    @param:
    @return: 
    '''
    rows = []
    count = 0
    for i in input:
        tmp = []
        i[0] = str(i[0]).split('/')[-1]
        j = str(i[0]).split('_')
        tmp.append(j[2])
        tmp.append(i[0])
        if j[1] not in problemNum:
            problemNum[j[1]] = 0
            type_data[j[1]] = count
            problemNumConv[count] = j[1]
            count += 1
        tmp.append(type_data.get(j[1]))
        tmp.append('0')
        tmp.append(i[1])
        rows.append(tmp)
    rows.sort(key=takeEle)
    return getCount(rows)


def getCount(rows):
    '''
    @description: Count dataset
    @demands:
    @param: rows: dataset
    @return:
    '''
    input = rows

    studentNum = []

    count = 0
    for i in rows:
        if i[0] not in studentNum:
            studentNum.append(i[0])
            count = count + 1
    # print("student count:", count)
    for i in rows:
        problemNum[problemNumConv[i[2]]] += 1
    # print("problem num", problemNum)
    problemStu = {}
    index = 0
    for i in range(len(rows)):
        if index >= len(rows):
            break
        tmp = []
        tmpStr = rows[index][0]
        while index < len(rows) and tmpStr == rows[index][0]:
            if rows[index][2] not in tmp:
                tmp.append(rows[index][2])
                if problemNumConv[rows[index][2]] not in problemStu.keys():
                    problemStu[problemNumConv[rows[index][2]]] = 0
                problemStu[problemNumConv[rows[index][2]]] += 1
            index += 1
    # print("student num for each problem", problemStu)

    problemC = {}
    problemE = {}
    for i in rows:
        if i[1][0] == 'c':
            if problemNumConv[i[2]] not in problemC.keys():
                problemC[problemNumConv[i[2]]] = 0
            problemC[problemNumConv[i[2]]] += 1
        else:
            if problemNumConv[i[2]] not in problemE.keys():
                problemE[problemNumConv[i[2]]] = 0
            problemE[problemNumConv[i[2]]] += 1
    # print("correct problem num:", problemC)
    # print("error problem num:", problemE)
    # print(problemNumConv)
    count = 0
    sum = 0
    for i in problemNumConv.values():
        # print('C-', count, ' ', problemStu[i],
        #       ' ', problemNum[i],
        #       ' ', problemC[i], end=' ')
        # if i in problemE:
        #     print(problemE[i])
        # else:
        #     print('0')
        count += 1
        sum += problemNum[i]
    # print('rows.len:', rows.__len__())

    return deleteCodes(input, problemNum)


MIN = 100  # the minimum number of problem


def deleteCodes(rows, problemNum):
    '''
    @description: Delete too few problems from dataset that less than 'MIN'
                  Sort probelms based on difficulty
    @demands:
    @param: 
    @return: dataset
    '''
    result = []
    deletePro = []
    for i in problemNum.keys():
        if problemNum[i] < MIN:
            deletePro.append(i)

    for j in rows:
        if j[1].split('_')[1] not in deletePro:
            result.append(j)

    problems = {}
    count = 0
    for i in result:
        if i[2] not in problems.keys():
            problems[i[2]] = count
            count += 1

    for i in result:
        i[2] = problems[i[2]]

    return result


def main():
    rows = []
    with open(filename, 'r', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            rows.append(row)

    rows = getGroup(rows)

    index = 0
    with open(file_train_out, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        while index < len(rows) * 0.8:
            index = write_group(rows, index, writer)

    with open(file_test_out, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        while index < len(rows):
            index = write_group(rows, index, writer)


if __name__ == '__main__':
    main()
