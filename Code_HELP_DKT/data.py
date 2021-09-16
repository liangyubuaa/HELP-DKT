import csv
import random


def load_data(fileName):
    rows = []
    max_skill_num = 0
    max_num_problems = 0
    with open(fileName, "r", encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    index = 0
    print("the number of rows is " + str(len(rows)))
    tuple_rows = []
    # turn list to tuple
    while (index < len(rows) - 1):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index + 1]))
        if (problems_num <= 1):
            index = index + 3 + problems_num
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index + i] for i in range(int(rows[index][0]) + 3))
            tup = tuple(tup)
            '''
            tup = (rows[index], rows[index+1], rows[index+2],rows[index+3])
            '''
            tuple_rows.append(tup)
            index += int(rows[index][0]) + 3

        if (tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
    # shuffle the tuple
    random.shuffle(tuple_rows)
    print("The number of students is ", len(tuple_rows))
    print("Finish reading data")
    print('max_num_probles:', max_num_problems,
          'max_skill_num:', max_skill_num + 1)
    return tuple_rows, max_num_problems, max_skill_num + 1
