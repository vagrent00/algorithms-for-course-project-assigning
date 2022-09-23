import numpy as np
import pandas as pd
from assign import munkres

# data: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/project.csv', header=None)))
# print(data)
# print(type(data[1, 5]))
# print(data[1, :])
# (row_num, col_num) = np.shape(data)
# max = [5] * 17
# print(zip(np.arange(10), np.arange(10)))
# index_student = np.arange(62)
# matrix = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
# max = [1, 1, 1]
# print(munkres(matrix, max, 3, 3)[1])
discard_info = [1, 2, 3, 4, 1, 5, 1]
print(discard_info.index(max(discard_info)))
