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
    original_matrix=data.astype('int')

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

def Post_Process(matched_matrix):
    result = np.zeros((row_num, 3))
    for i in range(0, row_num):
        for j in range(0, col_num):
            if matched_matrix[i][j] == 1:
                result[i][0] = i + 1
                result[i][1] = j + 1
                result[i][2] = original_matrix[i][j]
    result = pd.DataFrame(result)
    result.to_csv("result.csv")

# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():
    data=Load_Data()
    Process_Data(data)
    #initialize()
    max=[5]*col_num
    print(max)
    matched_matrix=munkres(matrix,max,row_num,col_num)
    #Output_Data(matched_matrix)
    Post_Process(matched_matrix)

if __name__ == '__main__':
    main()


