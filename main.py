# -------------------------------------------------
# Import packages
# -------------------------------------------------

from math import ceil
import numpy as np
import pandas as pd
from assign import munkres
from munkres import Munkres


# -------------------------------------------------
# Load and preprocess data
# -------------------------------------------------

def Process_Data():
    # TODO: convert the original excel file into these three csv files
    # load cost matrix, student information matrix and project information matrix from 3 csv files in data folder
    data: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/data.csv', header=None)))
    student_info: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/student.csv', header=None)))
    project_info: np.ndarray = np.array(pd.DataFrame(pd.read_csv('data/project.csv', header=None)))
    row_num, col_num = data.shape
    # set unfilled cost as 100 (high)
    for row in range(row_num):
        for col in range(col_num):
            if data[row][col] == ' ':
                data[row][col] = 100
    # set unfilled project requirements as 0 (no requirement)
    for row in range(project_info.shape[0]):
        for col in range(project_info.shape[1]):
            if not 0 <= project_info[row][col] <= 100:
                project_info[row][col] = 0
    # the original cost matrix that will be referred to throughout the program
    original_matrix0 = data.astype('int')
    # square the preference value to punish low priority, and the convex transformation will make the deviation small. e.g. 3^2 + 4^2 < 2^2 + 5^2
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
        # deal with no matching conditions
        if matched_student[row] == -1:
            preference = 1000
        else:
            preference = original_matrix[row][matched_student[row]]
        result.append([number, project, preference])
    # print("result", [a[2] for a in result])
    result = pd.DataFrame(result)
    # the result shows the projects students are assigned to and the preference
    result.to_csv("result.csv")
    matched_per_project = []
    for col in range(col_num):
        matched_per_project.append(matched_student.count(col))
    # print("matched_per_project", matched_per_project)


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
    :param i: the ith loop. special cases are there that i is set to be 100, which means that no projects will be discarded
    :param original_matrix0: the original matrix without any process, used for final result
    :param discarded_projects: the column index of discarded projects
    :param tolerance:int, matching with preference greater than it will be removed
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
            if matched_matrix[row][col] == 1 and original_matrix0[dic_student[row]][dic_project[col]] > tolerance:
                matched_matrix[row][col] = 0

    # projects with enrollment below the lower limit may be discarded, and the lower limit is increased every three turns during the first-round selection
    # some special rounds (marked with i = 100) don't need to discard projects
    if i >= 100:
        lower_limit = -1
    else:
        lower_limit = i // 3

    # -------------------------------------------------
    # 2. For projects that are full of people, assign students to
    # corresponding projects and remove the projects from further assignment.
    # If there exist projects that have enrollment fewer than the lower limit,
    # discard the project with the least enrollment and release students to the next turn for matching.
    # -------------------------------------------------

    discard_num = 0  # the number of projects whose enrollment is less than the lower limit
    discard_col = []  # the index of project discard candidates
    discard_info = []  # number of enrollment of potential discards

    # traverse every project in the matching process
    count_project = len(matched_per_project)
    for col in range(count_project):
        # the project is full of people
        if matched_per_project[col] + matched_student.count(dic_project[col]) == 5:
            count_project -= 1

            for row in range(len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # register the student, so that he will not enter next assignment
                    matched_student[dic_student[row]] = dic_project[col]
            # the project cannot enroll any more students
            max[col] = 0
        # the project has enough students
        elif matched_per_project[col] + matched_student.count(dic_project[col]) == 4:
            # the project can enroll at most one student
            for row in range(len(dic_student)):
                if matched_matrix[row][col] == 1:
                    # register those matched students
                    matched_student[dic_student[row]] = dic_project[col]
            max[col] = 1
        # the project has too fewer students, so that it will be marked as discard candidate and may be discarded if it has the fewest enrollment
        elif matched_per_project[col] + matched_student.count(dic_project[col]) <= lower_limit:
            discard_num += 1
            discard_col.append(col)
            discard_info.append(matched_per_project[col] + matched_student.count(dic_project[col]))

        elif i >= 100:  # special round (major assignment round) needs to register students even if the projects have no enough students
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
    # print("dic_student", new_dic_student)
    # print("dic_project", new_dic_project)
    # print("max", max)
    # print("matrix", new_matrix)
    return new_matrix, matched_student, max, count_student, count_project, new_dic_student, new_dic_project, discarded_projects


def major_assignment(col_num, discarded_projects, project_info, row_num, student_info, original_matrix,
                     original_matrix0):
    """
    perform one round of assignments, so that except for the projects that are already discarded, all projects' major requirement will be satisfied.

    :param col_num: # total projects
    :param discarded_projects: the list of projects that are already discarded before entering this major assignment round
    :param project_info: matrix records the requirements of projects
    :param row_num: # total students
    :param student_info: matrix records the requirements of students
    :param original_matrix: complete matrix with doubled cost
    :param original_matrix0: complete original matrix
    :return:

    """
    # -------------------------------------------------
    # 1. take out the major-specified slot of projects and assign students to them
    # -------------------------------------------------

    total_major_requirement_per_project = []
    # record the index of the major-specified slot
    curr_slot = 0
    # the dict that records slot index with which original project it belongs to.
    slot_project_correspondence = {}
    # the cost matrix for major assignment, each row is the cost of one slot
    major_assignment_matrix = []
    # the index of projects that are not discarded
    remained_project = []

    # traverse through all remaining projects
    for col in range(col_num):
        if col not in discarded_projects:
            total_major_requirement_per_project.append(
                project_info[1, col] + project_info[2, col] + project_info[3, col])
            remained_project.append(col)
            # for one project, it may have 3 types of major-specified slots:
            for major in 0, 1, 2:
                # if this project requires slot of this major:
                if not project_info[major + 1, col] == 0:
                    # construct one slot for each requirement
                    for i in range(project_info[major + 1, col]):
                        # preference vector for this slot
                        row = [0] * row_num
                        for j in range(row_num):
                            # if major is not matched, the cost will be very high
                            if not student_info[j, 1] == major:
                                row[j] = 10000
                            # if major matched, cost is the same as original
                            else:
                                row[j] = original_matrix[j, col]
                        # add this preference vector to the cost matrix
                        major_assignment_matrix.append(row)
                        slot_project_correspondence[curr_slot] = col  # record the slot-project correspondence
                        curr_slot += 1

    major_assignment_matrix = np.array(major_assignment_matrix)
    # matched_matrix, matched_per_student = munkres(major_assignment_matrix.copy(), [1] * row_num,
    #                                               major_assignment_matrix.shape[0], major_assignment_matrix.shape[1])

    # conduct matching between slots and students, matched matrix's row is one slot, column is one student
    m = Munkres()
    indexes = m.compute(major_assignment_matrix.copy())
    matched_matrix = np.zeros(major_assignment_matrix.shape, dtype=int)
    for i, j in indexes:
        matched_matrix[i, j] = 1

    # rearrange the matching matrix into the usual format, each row is changed back to students and each column is changed back to projects
    formatted_matched_matrix = np.zeros((row_num, col_num - len(discarded_projects)), dtype=int)
    dic_project = {}

    for i in range(col_num - len(discarded_projects)):
        dic_project[i] = remained_project[i]

    for i in range(row_num):
        for j in range(major_assignment_matrix.shape[0]):
            if matched_matrix[j, i]:
                formatted_matched_matrix[i, remained_project.index(slot_project_correspondence[j])] = 1

    # print('total_major_requirement_per_project', total_major_requirement_per_project)

    # Use Update_matrix function to generate assignment information (matched_student, matrix to be assigned next time). i is set to 100 so that no projects will be discarded and students can enroll even if the enrollment is small
    # the return values are marked with 0 (e.g. matrix0) and be returned by this function so that user can see the assignment situation when only major-specified slots are assigned.
    matrix0, matched_student0, max0, count_student0, count_project0, dic_student0, dic_project0, discarded_projects0 = Update_matrix(
        [5] * (col_num - len(discarded_projects)),
        formatted_matched_matrix,
        total_major_requirement_per_project,
        [-1] * row_num,
        original_matrix,
        {index: index for index in range(row_num)},
        dic_project,
        row_num,
        col_num, 100,
        original_matrix0, discarded_projects, 10000)
    # print('matched_student0 in major assignment', matched_student0)
    # print('max0 in major assignment', max0)

    # assign other students to the usual slots after the major-specified slots are satisfied
    matched_matrix, matched_per_project = munkres(matrix0.copy(), max0, count_student0, count_project0)
    matrix, matched_student, max, count_student, count_project, dic_student, dic_project, discarded_projects = Update_matrix(
        max0.copy(),
        matched_matrix,
        matched_per_project,
        matched_student0.copy(),
        original_matrix,
        dic_student0,
        dic_project0,
        row_num,
        col_num, 100,
        original_matrix0, discarded_projects0, 9)

    # do the assignment once again, and this time one project may be discarded  ( i not equal to 100)
    matched_matrix, matched_per_project = munkres(matrix, max, count_student, count_project)
    matrix, matched_student, max, count_student, count_project, dic_student, dic_project, discarded_projects = Update_matrix(
        max,
        matched_matrix,
        matched_per_project,
        matched_student,
        original_matrix,
        dic_student,
        dic_project,
        row_num,
        col_num, 9,
        original_matrix0, discarded_projects, 5)
    # print('new turn')
    # print('discarded_projects', discarded_projects)

    return discarded_projects, matched_student, matrix0, matched_student0, max0, count_student0, count_project0, dic_student0, dic_project0, discarded_projects0


def offline_test(discarded_projects, matched_student, student_info, project_info, col_num, row_num):
    """
    test if the given matched_student result meet the requirement of offline students of projects

    :return:
    problematic_projects: the list of indexes of projects that fail to meet the requirement of offline students
    problematic_projects_students: list[list[int]], record the online students in projects lacking offline students
    is_offline_satisfied: bool
    """
    dic_project = {}
    remained_project = []
    index = 0
    problematic_projects = []
    problematic_projects_students = []

    # record the projects that are remained
    for i in range(col_num):
        if i not in discarded_projects:
            dic_project[index] = i
            remained_project.append(i)
            index += 1

    # traverse the remained projects
    for project in remained_project:
        # the list to record online students
        student_list = []
        num_offline = 0
        for i in range(row_num):
            if matched_student[i] == project:
                # if the student assigned to this project is offline, add 1 to offline number
                if student_info[i, 2] == 1:
                    num_offline += 1
                # else, record the online student
                else:
                    student_list.append(i)

        # if the project have no enough offline students
        if num_offline < project_info[4, project]:
            # print(f'project {project} has no enough offline student!')
            # append the project index and corresponding online students
            problematic_projects.append(project)
            problematic_projects_students.append(student_list)
            # print(student_list)

    if problematic_projects:
        is_offline_satisfied = False
    else:
        is_offline_satisfied = True

    return problematic_projects, problematic_projects_students, is_offline_satisfied


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

    # while the projects discarded is less than the expected number, keep assigning and discarding
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
        # print("matched_student:", matched_student)
        # print("discarded", discarded_projects)
        # print(i, "turn")

    # Output the result
    # Output(matched_student, original_matrix0, row_num, col_num)
    print("loose selection finished")

    # --------------------------------------
    # after the loose selection is finished, do the major assignment repeatedly until no projects will be discarded
    # --------------------------------------
    discard_num = len(discarded_projects)
    discarded_projects, matched_student, matrix0, matched_student0, max0, count_student0, count_project0, dic_student0, dic_project0, discarded_projects0 = major_assignment(col_num, discarded_projects, project_info, row_num, student_info, original_matrix, original_matrix0)

    while not len(discarded_projects) == discard_num:
        discard_num = len(discarded_projects)
        discarded_projects, matched_student, matrix0, matched_student0, max0, count_student0, count_project0, dic_student0, dic_project0, discarded_projects0 = major_assignment(col_num, discarded_projects, project_info, row_num, student_info, original_matrix, original_matrix0)

    # print('matched_student0', matched_student0)
    # print('matrix0', matrix0)
    # print('max0', max0)

    # --------------------------------
    # test if the result of major assignment satisfies the offline requirement
    # if it's not satisfied, increase the cost of online students to the projects lacking offline students and reassign them until the requirement is satisfied
    # --------------------------------

    problematic_projects, problematic_projects_students, is_offline_satisfied = offline_test(discarded_projects, matched_student, student_info, project_info, col_num, row_num)
    reassign_time = 0
    while not is_offline_satisfied and reassign_time <= 5:
        # traverse unsatisfied projects
        for i in range(len(problematic_projects)):
            # we don't want to reassign students assigned to major-specified slots, so the reassignment is based on matrix0, which is the cost matrix after major-slot assignment and before the other slot assignment in major_assignment function.
            for j in range(matrix0.shape[1]):
                # find the dissatisfied project in matrix0
                if dic_project0[j] == problematic_projects[i]:
                    for ii in range(matrix0.shape[0]):
                        if dic_student0[ii] in problematic_projects_students[i]:
                            # increase the cost of assigned online students
                            matrix0[ii, j] += 30

        # print(matrix0)
        # print(matched_student0)

        # do the assignment based on increased costs
        matched_matrix, matched_per_project = munkres(matrix0.copy(), max0, count_student0, count_project0)
        matrix, matched_student, max, count_student, count_project, dic_student, dic_project, discarded_projects = Update_matrix(
            max0,
            matched_matrix,
            matched_per_project,
            matched_student0,
            original_matrix,
            dic_student0,
            dic_project0,
            row_num,
            col_num, 100,
            original_matrix0, discarded_projects0, 10)

        problematic_projects, problematic_projects_students, is_offline_satisfied = offline_test(discarded_projects, matched_student, student_info, project_info, col_num, row_num)
        reassign_time += 1

    print('matched_student', matched_student)
    result = []
    unmatched_students = []
    mismatched_students = []
    switch_candidate = []

    for row in range(row_num):
        # deal with unmatching conditions
        if matched_student[row] == -1:
            preference = 1000
            unmatched_students.append(row)
        else:
            preference = original_matrix0[row][matched_student[row]]
        if 9 <= preference <= 999:
            mismatched_students.append(row)
        elif 4 <= preference <= 8:
            switch_candidate.append(row)
        result.append([row + 1, matched_student[row] + 1, preference])
    print("result", [a[2] for a in result])
    result = pd.DataFrame(result)
    # the result shows the projects students are assigned to and the preference
    result.to_csv("result.csv")
    matched_per_project = []
    for col in range(col_num):
        matched_per_project.append(matched_student.count(col))
    print("matched_per_project", matched_per_project)

    print(mismatched_students, unmatched_students, switch_candidate)

    # this step is to fill in unfilled projects if the preference is good
    for project in range(col_num):
        if 0 < matched_per_project[project] < 5:  # these projects have already satisfied the requirements, so any students are welcome
            for student in unmatched_students.copy():
                if original_matrix0[student][project] <= 10:
                    matched_student[student] = project
                    matched_per_project[project] += 1
                    unmatched_students.remove(student)
                    if original_matrix0[student][project] >= 4:
                        switch_candidate.append(student)
                    if matched_per_project[project] == 5:
                        break

    print("matched_per_project", matched_per_project)

    # if one project still can't enroll enough people, discard it
    for project in range(col_num):
        if 0 < matched_per_project[project] < 4:
            matched_per_project[project] = 0
            for student in range(row_num):
                if matched_student[student] == project:
                    matched_student[student] = -1
                    unmatched_students.append(student)
                    if student in switch_candidate:
                        switch_candidate.remove(student)

    # if some students are still not assigned, enumerate all discarded projects to choose
    print("matched_per_project", matched_per_project)
    print(mismatched_students, unmatched_students, switch_candidate)

    # num_target_project = ceil(len(unmatched_students) / 5)
    # discarded_projects = []
    # num_candidate = []
    #
    # for project in range(col_num):
    #     if matched_per_project[project] == 0:
    #         num = 0
    #         discarded_projects.append(project)
    #         for student in unmatched_students:
    #             if original_matrix0[student][project] <= 10:
    #                 num += 1
    #         print(num)
    #         num_candidate.append(num)
    #

    # TODO: original_matrix0 is not changed. matched_student and matched_per_project is up-to-date. major: student_info[student_index, 1], offline: student_info[student_index, 2], project major requirement for 0, 1, 2: project_info[1/2/3, project_index], offline requirement: project_info[4, project_index]


if __name__ == '__main__':
    main()
