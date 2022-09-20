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
    data = pd.read_csv("data/data.csv")  # header=None, how to get this data.csv
    data = pd.DataFrame(data)
    data = np.array(data)
    (row_num, col_num) = np.shape(data)
    return data, row_num, col_num


def Process_Data(data, row_num, col_num):
    for row in range(0, row_num):
        for col in range(0, col_num):
            if data[row][col] == ' ':
                data[row][col] = 100
    # same value currently, but used for different purposes
    original_matrix0 = data.astype('int')
    # square the preference value to punish low priority
    original_matrix = original_matrix0 ** 2
    student_info =
    project_info =
    return original_matrix.copy(), original_matrix0, original_matrix


# -------------------------------------------------
# Output data
# -------------------------------------------------

def Output(matched_student, original_matrix, row_num, col_num):
    result = []
    for row in range(row_num):
        number = row + 1
        project = matched_student[row] + 1
        # deal with unmatching conditions
        if matched_student[row] == -1:
            preference = 1000
        else:
            preference = original_matrix[row][matched_student[row]]
        result.append([number, project, preference])
    print("result", [a[2] for a in result])
    result = pd.DataFrame(result)
    # the result shows the projects students are assigned to and the preference
    result.to_csv("result.csv")
    matched_per_project = []
    for col in range(col_num):
        matched_per_project.append(matched_student.count(col))
    print("matched_per_project", matched_per_project)


# -------------------------------------------------
# update students and project to be matched
# -------------------------------------------------
def Update_matrix(max, matched_matrix, matched_per_project, matched_student, original_matrix, dic_student, dic_project,
                  row_num, col_num, i, original_matrix0):
    """
    process the result and assign students to projects after the matching process

    :param max: an array that records the maximum students that a project can still enroll
    :param matched_matrix: a 2D array where 1 represents a student successfully matches a project while 0 doesn't
    :param matched_per_project: an array about how many students match a certain project
    :param matched_student: an array about the project number that a certain student is enrolled
    :param original_matrix: the original students' preference matrix, used for updating matrix for the next turn
    :param dic_student: the dictionary to convert the order of students in the matching process to the original order
    :param dic_project: the dictionary to convert the order of projects in the matching process to the original order
    :param row_num: number of students
    :param col_num: number of projects
    :param i: the ith loop
    :param original_matrix0: the original matrix without any process, used for final result
    :return:
    new_matrix: updated students' preference matrix that is used in the next matching process
    matched_student: an array about the project number that a certain student is enrolled
    max: an array that records the maximum students that a project can still enroll
    count: the number of students that aren't enrolled in a project
    count_student: number of students remained to be matched in the next turn
    count_project: number of projects remained to be matched in the next turn
    new_dic_student: the updated dictionary of student order
    new_dic_project: the updated dictionary of project order
    """

    # -------------------------------------------------
    # 1. Some preparation work
    # -------------------------------------------------

    # eliminate those matching with low priority
    for row in range(len(dic_student)):
        for col in range(len(dic_project)):
            if matched_matrix[row][col] == 1 and original_matrix0[dic_student[row]][dic_project[col]] > 5:  # TODO: find the best hyperparameter
                matched_matrix[row][col] = 0

    # projects enrolled below the lower limit will be eliminated,
    # and the lower limit is increased every three turns
    lower_limit = 0

    if i < 12:
        lower_limit = i // 3  # TODO: when round exceeds 12, 可能会死循环(无意义地持续到15回合)，因为不会有project被discard。是不是把12之后的limit全部设成3比较好，并且第九轮就是3是不是有点早。

    # -------------------------------------------------
    # 2. For projects that are full of people, assign students to
    # corresponding projects and remove the projects.
    # For projects that have people fewer than the lower limit,
    # discard the projects and leave students to the next turn for matching.
    # -------------------------------------------------

    # reset the matrix to the original matrix
    matrix = original_matrix

    # traverse every project in the matching process
    count_project = len(matched_per_project)
    for col in range(0, len(matched_per_project)):
        if matched_per_project[col] + matched_student.count(dic_project[col]) == 5:  # TODO: dedicated to the upper limit 5 for each project
            # the project is full of people
            count_project -= 1

            for row in range(0, len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # the student has been enrolled, and can't be enrolled in other projects
                    matched_student[dic_student[row]] = dic_project[col]
            # the project cannot enroll any more students
            max[col] = 0
        elif matched_per_project[col] + matched_student.count(dic_project[col]) == 4:
            # the project can enroll at most one student
            for row in range(0, len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # enroll those matched students
                    matched_student[dic_student[row]] = dic_project[col]
            max[col] = 1
        elif matched_per_project[col] + matched_student.count(dic_project[col]) <= lower_limit:
            # Because the projects have too few people, they would be discarded.
            max[col] = 0
            count_project -= 1

    # -------------------------------------------------
    # 3. derive the matrix to be assigned
    # (pick up the students and projects remained)
    # -------------------------------------------------

    new_matrix = []
    help_student = []  # the rest students

    # remove students already assigned to a project
    for row in range(0, row_num):
        if matched_student[row] == -1:
            new_matrix.append(original_matrix[row].tolist())
            help_student.append(row)

    # remove projects full of people or those which have been discarded
    help_matrix = []  # temporarily save the matrix and finally assign to the new_matrix
    for col in range(col_num):
        if col in dic_project.values() and max[list(dic_project.keys())[list(dic_project.values()).index(col)]] >= 1:
            if not help_matrix:
                # deal with the special cases that it is the first project to add
                for row in new_matrix:
                    help_matrix.append([row[col]])
            else:
                for row in range(len(help_student)):
                    help_matrix[row].append(new_matrix[row][col])
    new_matrix = help_matrix
    new_matrix = np.array(new_matrix)

    # -------------------------------------------------
    # 4. update the new dictionary to record the
    # original order of students and projects remained
    # -------------------------------------------------

    # student dictionary
    index = np.arange(len(help_student))
    new_dic_student = {order: student for order, student in zip(index, help_student)}

    # project dictionary
    help_project = []
    for col in range(0, len(matched_per_project)):
        if max[col] >= 1:
            help_project.append(dic_project[col])
    index2 = np.arange(len(help_project))
    new_dic_project = {order: project for order, project in zip(index2, help_project)}

    # remove the discarded projects in max(projects with max 0)
    max = list(filter((0).__ne__, max))

    # count the number of students to be matched in the next turn
    count_student = 0
    for row in range(0, row_num):
        if matched_student[row] == -1:
            count_student += 1

    # -------------------------------------------------
    # 5. Output some results
    # -------------------------------------------------

    Output(matched_student, original_matrix0, row_num, col_num)
    print("dic_student", new_dic_student)
    print("dic_project", new_dic_project)
    print("max", max)
    print("matrix", new_matrix)
    return new_matrix, matched_student, max, count_student, count_project, new_dic_student, new_dic_project


# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():
    # preprocess the data
    data, row_num, col_num = Load_Data()
    matrix, original_matrix0, original_matrix = Process_Data(data, row_num, col_num)

    # initialize some variables
    max = [5] * col_num  # [5, 5, ...]
    matched_student = [-1] * row_num
    count_student = row_num
    count_project = col_num
    dic_student = {index: index for index in range(row_num)}  # {0: 0, 1: 1, 2: 2, ...}
    dic_project = {index: index for index in range(col_num)}
    i = 0  # record the number of turns

    # the main loop
    while -1 in matched_student and i < 15:
        # first match students with projects by the munkres algorithm
        matched_matrix, matched_per_project = munkres(matrix, max, count_student, count_project)
        # then process the matching result.
        # assign students to corresponding project, discard some projects,
        # and leave the rest to the next matching process
        matrix, matched_student, max, count_student, count_project, dic_student, dic_project = Update_matrix(max, matched_matrix, matched_per_project, matched_student, original_matrix, dic_student, dic_project, row_num, col_num, i, original_matrix0)

        i += 1
        print("matched_student:", matched_student)
        print(i, "turn")

    # Output the result
    Output(matched_student, original_matrix0, row_num, col_num)


if __name__ == '__main__':
    main()
