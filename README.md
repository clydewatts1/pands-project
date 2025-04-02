# pands-project
ATU Data Analytics Post Grad Diploma - Project

## Project Overview

### Project : Iris Data Analysis
### Author : Clyde Watts
### Date : 2025-04-01
### Course : ATU Post Grad Diploma in Data Analytics
### Module : Principles of Data Analytics
### Instructor : Andrew Beatty
### Description : This project is an analysis of the iris dataset. The iris dataset is a tiny dataset which is used for projects , this one included. The iris dataset consists of 4 features ( properties) and 1 target . The 4 features are the petal length and width , and the sepal length and width ( no idea what a sepal is). The target is the iris flower type , setosa, versicolor or virginica. There is a sample of 50 of each flower species ( target). The features are all measured in cm , and the target is a varchar string of the species.


## Flowchart

```mermaid
flowchart TD
    A[Start] --> B[Parse Arguments]
    B --> C{Write Config?}
    C -->|Yes| D[Write Config to File]
    D --> E[Exit]
    C -->|No| F{Load Config File?}
    F -->|Yes| G[Load Config File]
    F -->|No| H[Use Default Config]
    G --> I{Source Path Exists?}
    H --> I
    I -->|No| J[Log Error: Source Path Missing]
    J --> K[Exit]
    I -->|Yes| L{Target Path Exists?}
    L -->|No| M[Log Error: Target Path Missing]
    M --> K
    L -->|Yes| N[Load Data]
    N --> O{Data Loaded Successfully?}
    O -->|No| P[Log Error: Data Load Failed]
    P --> K
    O -->|Yes| Q[Convert to Metrics DataFrame]
    Q --> R{Metrics Conversion Successful?}
    R -->|No| S[Log Error: Metrics Conversion Failed]
    S --> K
    R -->|Yes| T[Generate Summary Statistics]
    T --> U{Summary Generated Successfully?}
    U -->|No| V[Log Error: Summary Generation Failed]
    V --> K
    U -->|Yes| W[Generate Report]
    W --> X{Report Generated Successfully?}
    X -->|No| Y[Log Error: Report Generation Failed]
    Y --> K
    X -->|Yes| Z[Generate Histogram]
    Z --> AA{Histogram Generated Successfully?}
    AA -->|No| AB[Log Error: Histogram Generation Failed]
    AB --> K
    AA -->|Yes| AC[Generate Scatter Plot]
    AC --> AD{Scatter Plot Generated Successfully?}
    AD -->|No| AE[Log Error: Scatter Plot Generation Failed]
    AE --> K
    AD -->|Yes| AF[Generate Box Plot]
    AF --> AG{Box Plot Generated Successfully?}
    AG -->|No| AH[Log Error: Box Plot Generation Failed]
    AH --> K
    AG -->|Yes| AI[Generate Box Plot II]
    AI --> AJ{Box Plot II Generated Successfully?}
    AJ -->|No| AK[Log Error: Box Plot II Generation Failed]
    AK --> K
    AJ -->|Yes| AL[Generate Boxen Plot]
    AL --> AM{Boxen Plot Generated Successfully?}
    AM -->|No| AN[Log Error: Boxen Plot Generation Failed]
    AN --> K
    AM -->|Yes| AO[Generate Violin Plot]
    AO --> AP{Violin Plot Generated Successfully?}
    AP -->|No| AQ[Log Error: Violin Plot Generation Failed]
    AQ --> K
    AP --> AR[Log: Analysis Completed Successfully]
    AR --> AS[End]
```

## Files

| Directory     | File Name             | Description                            | Notes                         |
|--------------:|----------------------:|---------------------------------------:|-------------------------------|
| .             | analysis.py           | Primary project python script          |                               |
| .             | anaysis_notebook.ipynb| Project Supporting Notebook            |                               |
| .             | Index                 | Iris Index                             | Source                        |
| .             | iris.data             | Iris Data - CSV                        | CSV - no bom , linux          |
| .             | iris.names            | Iris Names                             | Iris Data - Names             |
| .             | bezdekItis.data       | Iris Data - CSV                        | CSV - no bom , linux          |
| .             | README.md             | This file                              |                               |
| .             | analysis_report.txt   | Analysis Report                        |                               | 
| .             | analysis.log          | Script Log file                        |                               |   

### File : iris.data file format

The iris dataset is a tiny dataset which is used for projects , this one included. The iris dataset consists of 4 features ( properties) and 1 target . The 4 features are the petal length and width , and the sepal length and width ( no idea what a sepal is). The target is the iris flower type , setosa, versicolor or virginica. There is a sample of 50 of each flower species ( target). The features are all measured in cm , and the target is a varchar string of the species.




| Attribute Name | Type    | Description                                                                 |
|----------------|---------|-----------------------------------------------------------------------------|
| sepal_length   | float   | Length of the sepal in centimeters                                          |
| sepal_width    | float   | Width of the sepal in centimeters                                           |
| petal_length   | float   | Length of the petal in centimeters                                          |
| petal_width    | float   | Width of the petal in centimeters                                           |
| class          | string  | Class of the iris flower (Iris-setosa, Iris-versicolor, Iris-virginica)     |
## References

- Iris Dataset: https://archive.ics.uci.edu/dataset/53/iris
- Iris Dataset: Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
- https://www.geeksforgeeks.org/multi-plot-grid-in-seaborn/   : Used to create the multi-plot grid
- https://www.cuemath.com/geometry/area-of-an-ellipse/
- datacamp.com : The data analytics course I am currently taking
- https://www.statology.org/seaborn-table/ : Used to create the table to the side of a table
- https://seaborn.pydata.org/generated/seaborn.catplot.html : Used to create the catplot - boxplot,violinplot,boxenplot
- https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot : Used to create the violinplot
- https://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot : Used to create the boxenplot
- Google Gemini : What is a violin plot?
- Google Gemini : What is a boxen plot?
- kmeans clustering: https://www.datacamp.com/tutorial/k-means-clustering-python?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720821&utm_adgroupid=157156374951&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=733936221293&utm_targetid=aud-1832882613722:dsa-2218886984380&utm_loc_interest_ms=&utm_loc_physical_ms=1007877&utm_content=&accountid=9624585688&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&gad_source=1&gclid=Cj0KCQjwv_m-BhC4ARIsAIqNeBtcqQxamZLbu_HZzz-KqeYXnvGMhbiqEAkhefWZntcQgx3jKP1Yy2IaAuKgEALw_wcB