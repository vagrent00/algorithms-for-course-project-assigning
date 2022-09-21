# -------------------------------------------------
# Import packages
# -------------------------------------------------
import numpy as np
import pandas as pd
from assign import munkres


# -------------------------------------------------
# Load and preprocess data
# -------------------------------------------------

def Process_Data():
    # TODO: convert the original excel file into these three csv files
    data: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/data.csv')))
    student_info: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/student.csv', header=None)))
    project_info: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/project.csv', header=None)))
    row_num, col_num = data.shape
    for row in range(row_num):
        for col in range(col_num):
            if data[row][col] == ' ':
                data[row][col] = 100
    for row in range(project_info.shape[0]):
        for col in range(project_info.shape[1]):
            if not 0 <= project_info[row][col] <= 100:
                project_info[row][col] = 0
    # same value currently, but used for different purposes
    original_matrix0 = data.astype('int')
    # square the preference value to punish low priority
    original_matrix = original_matrix0 ** 2
    student_info = student_info.astype('int')
    project_info = project_info.astype('int')
    return original_matrix.copy(), original_matrix0, original_matrix, student_info, project_info, row_num, col_num


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
                  row_num, col_num, i, original_matrix0, discarded_projects, tolerance):
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
    :param discarded_projects: the column index of discarded projects
    :param tolerance: the highest cost that will not be removed
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
            if matched_matrix[row][col] == 1 and original_matrix0[dic_student[row]][dic_project[col]] > tolerance:  # TODO: find the best hyperparameter
                matched_matrix[row][col] = 0

    # projects enrolled below the lower limit will be eliminated,
    # and the lower limit is increased every three turns
    lower_limit = 0

    if i < 12:
        lower_limit = i // 3
    elif i >= 100:
        lower_limit = -1  # some special round don't need to discard projects

    # -------------------------------------------------
    # 2. For projects that are full of people, assign students to
    # corresponding projects and remove the projects.
    # For projects that have people fewer than the lower limit,
    # discard the projects and leave students to the next turn for matching.
    # -------------------------------------------------

    discard_num = 0  # the number of projects whose enrollment is less than the lower limit
    discard_col = []  # the column number of potential discards
    discard_info = []  # enrollment of potential discards

    # traverse every project in the matching process
    count_project = len(matched_per_project)
    for col in range(count_project):
        if matched_per_project[col] + matched_student.count(dic_project[col]) == 5:
            # the project is full of people
            count_project -= 1

            for row in range(len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # the student has been enrolled, and can't be enrolled in other projects
                    matched_student[dic_student[row]] = dic_project[col]
            # the project cannot enroll any more students
            max[col] = 0
        elif matched_per_project[col] + matched_student.count(dic_project[col]) == 4:
            # the project can enroll at most one student
            for row in range(len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # enroll those matched students
                    matched_student[dic_student[row]] = dic_project[col]
            max[col] = 1
        elif matched_per_project[col] + matched_student.count(dic_project[col]) <= lower_limit:
            # Because the projects have too few people, they would be discarded.
            discard_num += 1
            discard_col.append(col)
            discard_info.append(matched_per_project[col] + matched_student.count(dic_project[col]))

        elif i >= 100:  # special round need to register students even if the projects are not full
            for row in range(len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # enroll those matched students
                    matched_student[dic_student[row]] = dic_project[col]
                    max[col] -= 1
            if max[col] == 0:
                count_project -= 1

    # discarding project
    if discard_num == 1:
        max[discard_col[0]] = 0
        count_project -= 1
        discarded_projects.append(dic_project[discard_col[0]])

    elif discard_num > 1:
        max[discard_col[discard_info.index(min(discard_info))]] = 0
        count_project -= 1
        discarded_projects.append(dic_project[discard_col[discard_info.index(min(discard_info))]])

    # -------------------------------------------------
    # 3. derive the matrix to be assigned
    # (pick up the students and projects remained)
    # -------------------------------------------------

    new_matrix = []
    help_student = []  # the rest students

    # remove students already assigned to a project
    for row in range(row_num):
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
    return new_matrix, matched_student, max, count_student, count_project, new_dic_student, new_dic_project, discarded_projects


# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():
    # preprocess the data
    matrix, original_matrix0, original_matrix, student_info, project_info, row_num, col_num = Process_Data()

    # initialize some variables
    max = [5] * col_num  # [5, 5, ...]
    matched_student = [-1] * row_num
    count_student = row_num
    count_project = col_num
    dic_student = {index: index for index in range(row_num)}  # {0: 0, 1: 1, 2: 2, ...}
    dic_project = {index: index for index in range(col_num)}
    i = 0  # record the number of turns
    target_discard_num = col_num - row_num // 4  # the number of project to be discarded firstly
    discarded_projects = []  # record the column index of discarded projects

    # the main loop
    while len(discarded_projects) < target_discard_num:
        # first match students with projects by the munkres algorithm
        matched_matrix, matched_per_project = munkres(matrix, max, count_student, count_project)
        # then process the matching result.
        # assign students to corresponding project, discard some projects,
        # and leave the rest to the next matching process
        matrix, matched_student, max, count_student, count_project, dic_student, dic_project, discarded_projects = Update_matrix(
            max,
            matched_matrix,
            matched_per_project,
            matched_student,
            original_matrix,
            dic_student,
            dic_project,
            row_num,
            col_num, i,
            original_matrix0, discarded_projects, 5)

        i += 1
        print("matched_student:", matched_student)
        print("discarded", discarded_projects)
        print(i, "turn")

    # Output the result
    Output(matched_student, original_matrix0, row_num, col_num)
    print("loose selection finished")

    # assign students to the major requirements of projects
    total_major_requirement_per_project = []
    curr_slot = 0
    slot_project_correspondence = {}
    major_assignment_matrix = []
    remained_project = []

    for col in range(col_num):
        if col not in discarded_projects:
            total_major_requirement_per_project.append(project_info[1, col] + project_info[2, col] + project_info[3, col])
            remained_project.append(col)
            for major in 0, 1, 2:
                if not project_info[major + 1, col] == 0:
                    for i in range(project_info[major + 1, col]):
                        row = [0] * row_num
                        for j in range(row_num):
                            if not student_info[j, 1] == major:
                                row[j] = 10000
                            else:
                                row[j] = original_matrix[j, col]
                        major_assignment_matrix.append(row)
                        slot_project_correspondence[curr_slot] = col
                        curr_slot += 1

    major_assignment_matrix = np.array(major_assignment_matrix)
    matched_matrix, matched_per_student = munkres(major_assignment_matrix, [1] * row_num, major_assignment_matrix.shape[0], major_assignment_matrix.shape[1])

    formatted_matched_matrix = np.zeros((row_num, col_num - len(discarded_projects)), dtype=int)
    dic_project = {}

    for i in range(col_num - len(discarded_projects)):
        dic_project[i] = remained_project[i]

    # format the new matched_matrix to the original format
    for i in range(row_num):
        for j in range(major_assignment_matrix.shape[0]):
            if matched_matrix[j, i]:
                formatted_matched_matrix[i, remained_project.index(slot_project_correspondence[j])] = 1

    print('total_major_requirement_per_project', total_major_requirement_per_project)
    matrix, matched_student, max, count_student, count_project, dic_student, dic_project, discarded_projects = Update_matrix(
        [5] * (col_num - len(discarded_projects)),
        formatted_matched_matrix,
        total_major_requirement_per_project,
        [-1] * row_num,
        original_matrix,
        {index: index for index in range(row_num)},
        dic_project,
        row_num,
        col_num, 100,
        original_matrix0, discarded_projects, 20)


if __name__ == '__main__':
    main()
