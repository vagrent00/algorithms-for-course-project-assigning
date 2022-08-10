# --------
# import packages
# --------
import numpy as np
import pandas as pd
max_per_project=5
original_matrix=[]
matrix=[[1,2,3],[2,4,6],[3,6,9],[2,1,3],[1,5,2],[6,8,5],[3,4,5]]
# student's choices
major_matrix=[1,1,2]
position_matrix=[1,1,0]
#global row_num,col_num,covered_row,covered_column,covered_matrix,primed_matrix,matched_matrix,matched_per_student,matched_per_project,primed_uncovered_pair
row_num=0
col_num=0
covered_row=np.zeros(row_num,dtype=int)
covered_column=np.zeros(col_num,dtype=int)
covered_matrix=np.zeros((row_num,col_num),dtype=int)
primed_matrix=np.zeros((row_num,col_num),dtype=int)
matched_matrix=np.zeros((row_num,col_num),dtype=int)
matched_per_student=np.zeros(row_num,dtype=int)
matched_per_project=np.zeros(col_num,dtype=int)
primed_uncovered_pair=[-1,-1]
# the tempory matching

def Load_Data():
    data=pd.read_csv("data.csv")
    data=pd.DataFrame(data)
    data=np.array(data)
    global row_num,col_num
    (row_num,col_num)=np.shape(data)
    return data

def Process_Data(data):
    #are_nan=np.isnan(data)
    #data[are_nan]=100
    for row in range(0,row_num):
        for col in range(0,col_num):
            if data[row][col]==' ':
                data[row][col]=100
    global matrix, original_matrix
    matrix=data.astype('int')
    original_matrix=matrix

def initialize():
    global row_num,col_num,covered_row,covered_column,covered_matrix,primed_matrix,matched_matrix,matched_per_student,matched_per_project,primed_uncovered_pair
    covered_row = np.zeros(row_num, dtype=int)
    covered_column = np.zeros(col_num, dtype=int)
    covered_matrix = np.zeros((row_num, col_num), dtype=int)
    primed_matrix = np.zeros((row_num, col_num), dtype=int)
    matched_matrix = np.zeros((row_num, col_num), dtype=int)
    matched_per_student = np.zeros(row_num, dtype=int)
    matched_per_project = np.zeros(col_num, dtype=int)

def Output_Data(data):
    print("result", np.multiply(original_matrix, matched_matrix))
    data=pd.DataFrame(data)
    data=pd.DataFrame( matched_matrix)
    data.to_csv("result.csv")

def Cover_Row(row):
    global covered_matrix, covered_row
    for col in range(0, col_num):
        covered_matrix[row][col] = 1
    covered_row[row]=1

def Cover_Col(col):
    global covered_matrix, covered_column
    for row in range(0, row_num):
        covered_matrix[row][col] = 1
    covered_column[col]=1

def Uncover_Row(row):
    global covered_matrix, covered_row
    for col in range(0, col_num):
        covered_matrix[row][col] = 0
    covered_row[row]=0

def Uncover_Column(col):
    global covered_matrix, covered_column
    for row in range(0, row_num):
        covered_matrix[row][col] = 0
    covered_column[col]=0

def Find_Primed():
    primed_loc=[]
    for row in range(0,row_num):
        for col in range(0,col_num):
            if primed_matrix[row][col]==1:
                primed_loc.append([row,col])
    return primed_loc

def Find_Starred_Row(row):
    for col in range(0,col_num):
        if matched_matrix[row][col]==1:
            return col
    return -1

def Find_Primed_Column(col):
    for row in range(0,row_num):
        if primed_matrix[row][col]==1:
            return row
    print("Error occurs.")
    return -1

def Clear_Notation():
    global primed_matrix, covered_matrix, covered_column, covered_matrix, primed_matrix
    primed_matrix = np.zeros((row_num, col_num), dtype=int)
    covered_row = np.zeros(row_num, dtype=int)
    covered_column = np.zeros(col_num, dtype=int)
    covered_matrix = np.zeros((row_num, col_num), dtype=int)
    primed_uncovered_pair=[-1,-1]

def Noncovered_Zero():
    for row in range(0,row_num):
        for col in range(0,col_num):
            if covered_matrix[row][col]==0 and matrix[row][col]==0:
                return True
    return False

def Step1():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    for col in range(0,col_num):
        min=matrix[0][col]
        for row in range(0,row_num):
            if (matrix[row][col]<min):
                min=matrix[row][col]
        for row in range(0,row_num):
            matrix[row][col]-=min
    print(1)
    return 2

def Step2():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    for row in range(0,row_num):
        for col in range(0,col_num):
            if matrix[row][col]==0 and matched_per_student[row]==0 and matched_per_project[col]<max_per_project:
                matched_matrix[row][col]=1
                matched_per_student[row]+=1
                matched_per_project[col]+=1
    print(2)
    return 3



def Step3():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    print(matched_matrix)
    for row in range(0,row_num):
        for col in range(0,col_num):
            if matched_matrix[row][col]==1:
                Cover_Row(row)
    if np.sum(matched_per_student)==row_num:
        step = 7
    else:
        step = 4
    print(3)
    return step

def Step4():
    #print(matrix)
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    print(4)

    while(Noncovered_Zero()):
        for row in range(0,row_num):
            for col in range(0,col_num):
                if matrix[row][col]==0 and covered_matrix[row][col]==0:
                    primed_matrix[row][col]=1
                    if matched_per_project[col]<max_per_project:
                        primed_uncovered_pair=[row,col]
                        return 5
                    else:
                        for row in range(0,row_num):
                            if matched_matrix[row][col]==1:
                                Uncover_Row(row)
                        Cover_Col(col)
    Output_Data(matched_matrix)
    return 6

def Step5():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    print("matrix",matrix)
    print("cover",covered_matrix)
    print("prime",primed_uncovered_pair)
    star_series=[]
    unstar_series=[]
    row_loc=primed_uncovered_pair[0]
    col_loc=primed_uncovered_pair[1]
    star_series.append([row_loc,col_loc])
    while(Find_Starred_Row(row_loc)!=-1):
        col = Find_Starred_Row(row_loc)
        unstar_series.append([row_loc,col])
        row_loc=Find_Primed_Column(col)
        star_series.append([row_loc,col])
    for loc in star_series:
        row=loc[0]
        col=loc[1]
        matched_matrix[row, col] = 1
        matched_per_student[row] += 1
        matched_per_project[col] += 1
    for loc in unstar_series:
        row=loc[0]
        col=loc[1]
        matched_matrix[row, col] = 0
        matched_per_student[row] -= 1
        matched_per_project[col] -= 1
    Clear_Notation()
    print(5)
    return 3

def Step6():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    min=100
    for row in range(0, row_num):
        for col in range(0, col_num):
            if covered_matrix[row][col]==0 and matrix[row][col]<min:
                min=matrix[row][col]
    for row in range(0, row_num):
        if covered_row[row]==0:
            for col in range(0, col_num):
                matrix[row][col]-=min
    for col in range(0, col_num):
        if covered_column[col]==1:
            for row in range(0, row_num):
                matrix[row][col]+=min
    print(6)
    return 4

def munkres():
    whether_Continue = True
    step = 1
    while (whether_Continue):
        try:
            Func=steps[step]
            step=Func()
        except KeyError:
            whether_Continue=False
steps = { 1: Step1,
          2: Step2,
          3: Step3,
          4: Step4,
          5: Step5,
          6: Step6}

def main():
    data=Load_Data()
    Process_Data(data)
    initialize()
    munkres()
    Output_Data(matched_matrix)




if __name__ == '__main__':
    main()


