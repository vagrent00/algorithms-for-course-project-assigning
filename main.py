# -------------------------------------------------
# Import packages
# -------------------------------------------------
import numpy as np
import pandas as pd
from assign import munkres

# -------------------------------------------------
# Load and preprocess data
# -------------------------------------------------

def Load_Data():
    data=pd.read_csv("data.csv")
    data.to_csv("test.csv")
    data=pd.DataFrame(data)
    data=np.array(data)
    (row_num,col_num)=np.shape(data)
    return (data,row_num,col_num)

def Process_Data(data,row_num,col_num):
    #are_nan=np.isnan(data)
    #data[are_nan]=100
    for row in range(0,row_num):
        for col in range(0,col_num):
            if data[row][col]==' ':
                data[row][col]=100
    matrix=data.astype('int')
    original_matrix=data.astype('int')
    return (matrix,original_matrix)

    # -------------------------------------------------
    # Output data
    # -------------------------------------------------

'''
def Output_Data(data):
    # print("result", np.multiply(original_matrix, matched_matrix))
    data = pd.DataFrame(data)
    data = pd.DataFrame(np.multiply(original_matrix, matched_matrix))
    print(np.sum(np.sum(data)))
    print("each project has students", matched_per_project)
    data.to_csv("result.csv")
'''

def Post_Process(matched_matrix,original_matrix,row_num,col_num):
    result = np.zeros((row_num, 3))
    for row in range(0, row_num):
        for col in range(0, col_num):
            if matched_matrix[row][col] == 1:
                result[row][0] = row + 1
                result[row][1] = col + 1
                result[row][2] = original_matrix[row][col]
    result = pd.DataFrame(result)
    result.to_csv("result.csv")



# -------------------------------------------------
# update students and project to be matched
# -------------------------------------------------
def Update_matrix(max,matched_matrix,matched_per_project,matched_student,original_matrix,row_num,col_num,i):
    lower_limit=0
    if i<5:
        lower_limit=0
    elif i<10:
        lower_limit=1
    elif i<15:
        lower_limit=2
    elif i<20:
        lower_limit=3
    matrix=original_matrix
    for col in range(0,col_num):
        if matched_per_project[col]==5:
            for row in range(0,row_num):
                if matched_matrix[row][col]==1:
                    matched_student[row]=col
                    for col2 in range(0,col_num):
                        matrix[row][col2]=100
                matrix[row][col]=100
            max[col]=0
        elif matched_per_project[col]==4:
            for row in range(0,row_num):
                if matched_matrix[row][col]==1:
                    matched_student[row]=col
                    for col2 in range(0,col_num):
                        matrix[row][col2]=100
            max[col]=1
        elif matched_per_project[col]<=lower_limit:
            for row in range(0,row_num):
                matrix[row][col]=100
    count=0
    for row in range(0,row_num):
        if matched_student[row]==-1:
            count+=1
    return (matrix,matched_student,max,count)

# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():
    (data,row_num,col_num)=Load_Data()
    (matrix,original_matrix)=Process_Data(data,row_num,col_num)
    #initialize()
    max=[5]*col_num
    matched_student=[-1]*row_num
    count=row_num
    i=0
    while(np.prod([a+1 for a in matched_student])==0):
        (matched_matrix,matched_per_project)=munkres(matrix,max,row_num,col_num,count)
    #Output_Data(matched_matrix)
        (matrix,matched_student,max,count)=Update_matrix(max,matched_matrix,matched_per_project,matched_student,original_matrix,row_num,col_num,i)
        i+=1
        print(matched_student)
        print(i,"turn")
    Post_Process(matched_matrix,original_matrix,row_num,col_num)

if __name__ == '__main__':
    main()


