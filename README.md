# Stage-A Document Intelligence

## Installation

```bash
conda create -n docint python=3.11
conda activate docint
conda install -c conda-forge ghostscript
pip install -r requirements.txt
```

## Target

Create an artificial intelligence that searches in which table in the given pdf files has the desired information.  

#### Input

1. pdf files with only tables inside
2. the searching keywords

#### Output

**the hole table with desired information in it**

#### Example

The given pdf file:  
![image](https://github.com/Stage-A/Document-Intelligence/blob/main/images/example1.png)
Search query:  

```commandline
非監督式學習的應用
```

Output:
![image](https://github.com/Stage-A/Document-Intelligence/blob/main/images/example2.png)


#### Flow
![image](./images/flow.png)
