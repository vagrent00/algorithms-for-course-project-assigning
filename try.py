import numpy as np
import pandas as pd
from assign import munkres

# data = pd.read_csv("try.csv", header=None)
# print(data)
# data = pd.DataFrame(data)
# print(data)
# data = np.array(data)
# print(data)
# (row_num, col_num) = np.shape(data)
# max = [5] * 17
# print(zip(np.arange(10), np.arange(10)))
# index_student = np.arange(62)
matrix = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
max = [1, 1, 1]
print(munkres(matrix, max, 3, 3)[1])

