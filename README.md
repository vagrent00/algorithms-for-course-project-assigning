# Algorithm for Project Assigning (simple version)

### Purpose
This program is to help with instructors to assign students to different projects based on their ranking of perference to the projects.

### Feature
This version(v1.0) of program only find the best assignment results without constraints. If 
you want to satisfy projects' requirements about participants' major and offline status, please refer to v2.0.

### Usage
Place 1 csv files to data/ directory (you can refer to the existing examples):
+ data.csv: the preference matrix of students. Each row is one student and each column is one project.

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
