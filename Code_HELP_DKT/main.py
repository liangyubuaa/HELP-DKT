import os

dirPath = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # ....../code


def main():

    input_size, Qmatrix_size, inputConnectQmatrix, one_hot, inputMulQmatix = '20', '10', 'False', 'False', 'True'
    os.system("python " + dirPath + "/Code_HELP_DKT/RunModel.py -taskModel taskA -input_size %s -Qmatrix_size %s -inputConnectQmatrix %s -one_hot %s -inputQmatrixType P_Qmatrix -linearWithQmatrix False -QmatrixType P_Qmatrix -multiLinearLayers False -inputMulQmatix %s -masked False -set2zero False -subQmatrix False" %
              (input_size, Qmatrix_size, inputConnectQmatrix, one_hot, inputMulQmatix))

    input_size, Qmatrix_size, inputConnectQmatrix, one_hot, inputMulQmatix = '20', '10', 'False', 'False', 'True'
    os.system("python " + dirPath + "/Code_HELP_DKT/RunModel.py -taskModel taskB -input_size %s -Qmatrix_size %s -inputConnectQmatrix %s -one_hot %s -inputQmatrixType P_Qmatrix -linearWithQmatrix False -QmatrixType P_Qmatrix -multiLinearLayers False -inputMulQmatix %s -masked False -set2zero False -subQmatrix False" %
              (input_size, Qmatrix_size, inputConnectQmatrix, one_hot, inputMulQmatix))

    input_size, Qmatrix_size, inputConnectQmatrix, one_hot, inputMulQmatix = '20', '10', 'False', 'False', 'True'
    os.system("python " + dirPath + "/Code_HELP_DKT/RunModel.py -taskModel taskC -input_size %s -Qmatrix_size %s -inputConnectQmatrix %s -one_hot %s -inputQmatrixType P_Qmatrix -linearWithQmatrix False -QmatrixType P_Qmatrix -multiLinearLayers True -inputMulQmatix %s -masked True -set2zero True -subQmatrix True" %
              (input_size, Qmatrix_size, inputConnectQmatrix, one_hot, inputMulQmatix))


if __name__ == "__main__":
    main()
