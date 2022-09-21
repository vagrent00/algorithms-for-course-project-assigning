# -------------------------------------------------
# Import packages
# -------------------------------------------------
import numpy as np
import pandas as pd

# -------------------------------------------------
# Define global variables
# -------------------------------------------------
original_matrix = []
matrix = [[1, 2, 3], [2, 4, 6], [3, 6, 9], [2, 1, 3], [1, 5, 2], [6, 8, 5], [3, 4, 5]]
# student's choices
major_matrix = [1, 1, 2]
position_matrix = [1, 1, 0]
# global row_num,col_num,covered_row,covered_column,covered_matrix,primed_matrix,matched_matrix,matched_per_student,matched_per_project,primed_uncovered_pair
row_num = 0
col_num = 0
covered_row = np.zeros(row_num, dtype=int)
covered_column = np.zeros(col_num, dtype=int)
covered_matrix = np.zeros((row_num, col_num), dtype=int)
primed_matrix = np.zeros((row_num, col_num), dtype=int)
matched_matrix = np.zeros((row_num, col_num), dtype=int)
matched_per_student = np.zeros(row_num, dtype=int)
matched_per_project = np.zeros(col_num, dtype=int)
primed_uncovered_pair = [-1, -1]
max_per_project = np.zeros(col_num)


# the temporary matching


def initialize():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    covered_row = np.zeros(row_num, dtype=int)
    covered_column = np.zeros(col_num, dtype=int)
    covered_matrix = np.zeros((row_num, col_num), dtype=int)
    primed_matrix = np.zeros((row_num, col_num), dtype=int)
    matched_matrix = np.zeros((row_num, col_num), dtype=int)
    matched_per_student = np.zeros(row_num, dtype=int)
    matched_per_project = np.zeros(col_num, dtype=int)


# -------------------------------------------------
# Define help functions
# -------------------------------------------------
def Cover_Row(row):
    global covered_matrix, covered_row
    for col in range(0, col_num):
        covered_matrix[row][col] = 1
    covered_row[row] = 1


def Cover_Col(col):
    global covered_matrix, covered_column
    for row in range(0, row_num):
        covered_matrix[row][col] = 1
    covered_column[col] = 1


def Uncover_Row(row):
    global covered_matrix, covered_row
    for col in range(0, col_num):
        covered_matrix[row][col] = 0
    covered_row[row] = 0


def Uncover_Column(col):
    global covered_matrix, covered_column
    for row in range(0, row_num):
        covered_matrix[row][col] = 0
    covered_column[col] = 0


def Find_Primed():
    primed_loc = []
    for row in range(0, row_num):
        for col in range(0, col_num):
            if primed_matrix[row][col] == 1:
                primed_loc.append([row, col])
    return primed_loc


def Find_Starred_Row(row):
    for col in range(0, col_num):
        if matched_matrix[row][col] == 1:
            return col
    return -1


def Find_Primed_Column(col):
    for row in range(0, row_num):
        if primed_matrix[row][col] == 1:
            return row
    print("Error occurs.")
    return -1


def Clear_Notation():
    global primed_matrix, covered_matrix, covered_column, covered_matrix, primed_matrix
    primed_matrix = np.zeros((row_num, col_num), dtype=int)
    covered_row = np.zeros(row_num, dtype=int)
    covered_column = np.zeros(col_num, dtype=int)
    covered_matrix = np.zeros((row_num, col_num), dtype=int)
    primed_uncovered_pair = [-1, -1]


def Noncovered_Zero():
    for row in range(0, row_num):
        for col in range(0, col_num):
            if covered_matrix[row][col] == 0 and matrix[row][col] == 0:
                return True
    return False


# -------------------------------------------------
# Main steps
# -------------------------------------------------
def Step1():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    for col in range(0, col_num):
        min = matrix[0][col]
        for row in range(0, row_num):
            if matrix[row][col] < min:
                min = matrix[row][col]
        for row in range(0, row_num):
            matrix[row][col] -= min
    return 2


def Step2():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    for row in range(0, row_num):
        for col in range(0, col_num):
            if matrix[row][col] == 0 and matched_per_student[row] == 0 and matched_per_project[col] < max_per_project[
                col]:
                matched_matrix[row][col] = 1
                matched_per_student[row] += 1
                matched_per_project[col] += 1
    return 3


def Step3():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair

    for row in range(0, row_num):
        for col in range(0, col_num):
            if matched_matrix[row][col] == 1:
                Cover_Row(row)
    if np.sum(matched_per_student) == row_num:
        step = 7
    else:
        step = 4
    return step


def Step4():
    loop_num = 0
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair

    while Noncovered_Zero() and loop_num < 100:
        for row in range(0, row_num):
            for col in range(0, col_num):
                if matrix[row][col] == 0 and covered_matrix[row][col] == 0:
                    primed_matrix[row][col] = 1
                    if matched_per_project[col] < max_per_project[col]:
                        primed_uncovered_pair = [row, col]
                        return 5
                    else:
                        for row in range(0, row_num):
                            if matched_matrix[row][col] == 1:
                                Uncover_Row(row)
                        Cover_Col(col)
        loop_num += 1
    return 6


def Step5():
    loop_num = 0
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    star_series = []
    unstar_series = []
    row_loc = primed_uncovered_pair[0]
    col_loc = primed_uncovered_pair[1]
    star_series.append([row_loc, col_loc])
    while Find_Starred_Row(row_loc) != -1 and loop_num <= 100:
        col = Find_Starred_Row(row_loc)
        unstar_series.append([row_loc, col])
        row_loc = Find_Primed_Column(col)
        star_series.append([row_loc, col])
        loop_num += 1
    for loc in star_series:
        row = loc[0]
        col = loc[1]
        matched_matrix[row, col] = 1
        matched_per_student[row] += 1
        matched_per_project[col] += 1
    for loc in unstar_series:
        row = loc[0]
        col = loc[1]
        matched_matrix[row, col] = 0
        matched_per_student[row] -= 1
        matched_per_project[col] -= 1
    Clear_Notation()
    return 3


def Step6():
    global row_num, col_num, covered_row, covered_column, covered_matrix, primed_matrix, matched_matrix, matched_per_student, matched_per_project, primed_uncovered_pair
    min = 100
    for row in range(0, row_num):
        for col in range(0, col_num):
            if covered_matrix[row][col] == 0 and matrix[row][col] < min:
                min = matrix[row][col]
    for row in range(0, row_num):
        if covered_row[row] == 0:
            for col in range(0, col_num):
                matrix[row][col] -= min
    for col in range(0, col_num):
        if covered_column[col] == 1:
            for row in range(0, row_num):
                matrix[row][col] += min
    return 4


# -------------------------------------------------
# Core loops
# -------------------------------------------------
steps = {1: Step1,
         2: Step2,
         3: Step3,
         4: Step4,
         5: Step5,
         6: Step6}


def munkres(original_matrix, max, row, col):
    """
    the core munkres algorithm to match students with projects

    :param original_matrix: students' preference matrix
    :param max: an array that records the maximum students that a project can still enroll
    :param row: number of students
    :param col: number of projects
    :return:
    matched_matrix: a 2D array where 1 represents a student successfully matches a project while 0 doesn't
    matched_per_project: an array about how many students match a certain project
    """
    global matrix, row_num, col_num, max_per_project
    matrix = original_matrix
    max_per_project = max
    row_num = row
    col_num = col
    print("row", row)
    print("col", col)

    initialize()
    whether_Continue = True
    step = 1
    count = 0
    while whether_Continue:
        try:
            count += 1
            if count > 500:
                break
            Func = steps[step]
            step = Func()
        except KeyError:
            whether_Continue = False
    return matched_matrix, matched_per_project
