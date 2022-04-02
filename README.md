# HELP-DKT: an Interpretable Cognitive Model of How Students Learn Programming Based on Deep Knowledge Tracing

# About

This is an implementation of the HELP-DKT model, described in the following paper: HELP-DKT: an Interpretable Cognitive Model of How Students Learn Programming Based on Deep Knowledge Tracing (https://www.nature.com/articles/s41598-022-07956-0).

# Contributors

- **Yu Liang**: yliang@buaa.edu.cn 
- **Tianhao Peng**: pengtianhao@buaa.edu.cn
- **Meng Wang** 

_State Key Laboratory of Software Development Environment Admire Group, School of Computer Science and Engineering, Beihang University_

# Dataset
The original Python code files are compressed in the path: `./Data/Original_Codes.zip`. Please unzip it.

```
unzip ./Data/Original_Codes.zip
```

Our dataset includes 9119 source codes collected from a Python Programming Introductory course hosted in a MOOC platform for learning a variety of programming languages.

Each Python file in the dataset directory represents a student's submission, and the name of the file is organized as follows:

1. the result of submission(**'b'** for buggy, **'c'** for correct)
2. the challenge number of submission(e.g. '362', '371')
3. the student ID
4. the number of student's submissions on one challenge

## Example

Consider one submission 'b_362_27176_2'. The example represents:
1. 'b': buggy, the submission is error
2. '362': the challenge number is 362
3. '27173': the student ID is 27173
4. '2': the Python file is the student's second submission for the challenge

Another example 'c_449_27303_5':
1. 'c': correct, the submission is correct
2. '449': the challenge number is 449
3. '27303': the student ID is 27303
4. '5': the Python file is the student's 5th submission for the challenge

### Challenge

In our paper, we use 'C-1', 'C-2', ... , 'C-6' to represent the challenges instead of '362', '449',... , '472' for clarity. The following table shows the comparison relationship:

| Paper name | Original name |
| --- | --- |
| C-1 | 362 |
| C-2 | 371 |
| C-3 | 406 |
| c-4 | 417 |
| c-5 | 449 |
| c-6 | 472 |


# Program Vector Embeddings

see [Code_Program_Embeddings/README.md](./Code_Program_Embeddings/README.md)

# HELP-DKT Model

N.B. if you do not want to reproduce generating the program vector embeddings, you can start here to run the HELP-DKT model using the embedding results in [Program_Vector_Embeddings.CSV](Data/Program_Vector_Embeddings.CSV).

see [Code_HELP_DKT/README.md](./Code_HELP_DKT/README.md)

# Reference

```
@article{liang_help-dkt_2022,
	title = {{HELP}-{DKT}: an interpretable cognitive model of how students learn programming based on deep knowledge tracing},
	volume = {12},
	url = {https://doi.org/10.1038/s41598-022-07956-0},
	doi = {10.1038/s41598-022-07956-0},
	number = {1},
	journal = {Scientific Reports},
	author = {Liang, Yu and Peng, Tianhao and Pu, Yanjun and Wu, Wenjun},
	month = mar,
	year = {2022},
	pages = {4012},
}
```
