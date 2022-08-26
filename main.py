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
    '''
    process the result and assign students to projects after the matching process

    :param max: an array that records the maximum students that a project can still enroll
    :param matched_matrix: a 2D array where 1 represents a student successfully matches a project while 0 doesn't
    :param matched_per_project: an array about how many students match a certain project
    :param matched_student: an array about the project number that a certain student is enrolled
    :param original_matrix: the original students' preference matrix
    :param row_num: number of students
    :param col_num: number of projects
    :param i: the ith loop
    :return:
    new_matrix: updated students' preference matrix that is used in the next matching process
    matched_student: an array about the project number that a certain student is enrolled
    max: an array that records the maximum students that a project can still enroll
    count: the number of students that aren't enrolled in a project
    '''
    # projects enrolled below the lower limit will be eliminated,
    # and the lower limit is increased every five turns
    lower_limit=0
    if i<5:
        lower_limit=1
        # for debugging temporarily
    elif i<10:
        lower_limit=1
    elif i<15:
        lower_limit=2
    elif i<20:
        lower_limit=3

    # modify the value to 100 for those that can't be matched
    # when some projects are eliminated or the student is enrolled
    # in certain projects
    # reset the matrix to the original matrix
    matrix=original_matrix
    count_project=col_num
    for col in range(0,col_num):
        if matched_per_project[col]==5:
            # the project is full of people
            count_project-=1
            for row in range(0,row_num):
                if matched_matrix[row][col]==1:
                    matched_student[row]=col
                    # the student has been enrolled, and can't be enrolled in other projects
                    # for col2 in range(0,col_num):
                    #    matrix[row][col2]=100
               # matrix[row][col]=100
            # the project cannot enroll any more students
            max[col]=0
        elif matched_per_project[col]==4:
            # the project can enroll at most one student
            for row in range(0,row_num):
                if matched_matrix[row][col]==1:
                # enroll those matched students
                    matched_student[row]=col
                    # for col2 in range(0,col_num):
                    #    matrix[row][col2]=100
            max[col]=1
        # elif matched_per_project[col]<=lower_limit:
            # the project would be eliminated
        #    for row in range(0,row_num):
        #        matrix[row][col]=100

    # derive the matrix to be assigned
    new_matrix=[]
    help_student=[]
    for row in range(0,row_num):
        if matched_student[row]==-1:
            new_matrix.append(original_matrix[row])
            help_student.append(row)
    rest_matrix=[]
    help_matrix=[]
    for col in range(0,col_num):
        if matched_per_project[col]==5:
            for row in new_matrix:
                row=row.tolist()
                help_matrix=row[:col]
                help_matrix.extend(row[col+1:])
                rest_matrix.append(help_matrix)
    new_matrix=rest_matrix
    print(len(new_matrix))
    print(help_student)
    index=np.arange(len(help_student))
    print(index)

    count_student=0
    for row in range(0,row_num):
        if matched_student[row]==-1:
            count_student+=1
    return (new_matrix,matched_student,max,count_student,count_project)

# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():
    (data,row_num,col_num)=Load_Data()
    (matrix,original_matrix)=Process_Data(data,row_num,col_num)
    #initialize()
    max=[5]*col_num
    matched_student=[-1]*row_num
    count_student=row_num
    count_project=col_num
    i=0
    while(np.prod([a+1 for a in matched_student])==0):
        (matched_matrix,matched_per_project)=munkres(matrix,max,count_student,count_project)
    #Output_Data(matched_matrix)
        (matrix,matched_student,max,count_student,count_project)=Update_matrix(max,matched_matrix,matched_per_project,matched_student,original_matrix,row_num,col_num,i)
        i+=1
        print(matched_student)
        print(i,"turn")
    Post_Process(matched_matrix,original_matrix,row_num,col_num)

if __name__ == '__main__':
    main()


