Please Note: 

BERT code is obtained from the BERT's official GitHub, as mentioned in the paper:
https://github.com/google-research/bert


In addition to the existing files, for our task of sentiment analysis, we have added the following source code files: 

10fold.py
run_sosc.py
sosc_eval.py

-----------------------------------------------------------------------------------------------------------


Instructions for how to run the code.

1. Create a virtual environment in the root project directory with python 3.

2. In the root directory, pip install -r requirement.txt

3. In the root directory, create a directory named 'bert' for the source codes. Put all the source codes in that directory.

4. In the root directory, create a dataset and an output directory. Name the dataset directory as 'datasets' and the output directory as 'out' . 

5. Put the datasets, which are uploaded as 'NewData.csv' and 'StackOverflow_Original.csv' in the 'datasets' directory.
 
6. In the root directory, run the below command : 

python bert/10fold.py --datadir=datasets --out_dir=out

This will run the code and write 10 fold results in the output directory.

7. To change hyperparamaters in the BERT code,  you can modify the lines between 64-129 in the run_sosc.py file in the source code.


