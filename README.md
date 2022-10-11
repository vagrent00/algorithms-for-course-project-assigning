# Algorithm for Project Assigning (complete version)

### Purpose
This program is to help with instructors to assign students to different projects based on their ranking of perference to the projects. It can also be applied to other matching problems that minimize the costs.

### Feature
This version(v2.0) of program can satisfy projects' requirements about participants' major and offline status. If 
you only need to find the best assignment results without constraints, please refer to v1.0.

### Usage
Place 3 csv files to data/ directory (you can refer to the existing examples):
+ data.csv: the preference matrix of students. Each row is one student and each column is one project.
+ student.csv: the major & online/offline status of students. Each row is one student, the first column is index, the second column is major (ECE, ME, MSE) and the third column is online(0) or offline(1)
+ project.csv: the requirement from projects. From first row: index, requirement for the numer of ECE major students, ME major students, MSE major students and offline students.

### Output
The output goes to the result.csv file.
```css
    , 0,  1, 2
0   , 1, 14, 1
1   , 2, 17, 2
2   , 3, 13, 1
...
```
This means that the first student is assigned to 14th project with the first preference, and the second student is assigned to the 17th project with the second preference, and so on.
