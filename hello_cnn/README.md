# Hello CNN!
This is my first experiment with deep learning. this CNN model classified country based on person name, this CNN trained using chinese and russian names and can classified whether a person's name is chinese or russian name.

# Datasets
- There are 9800 chinese name, 9800 russian and 9800 arabic name (total 29400 person's name)
- This model trained with 23520 random mixed chinese and russian name the rest of 5880 is used for evaluation/testing data.
- train.csv and evaluation.csv extracted using gen_data.py (for parsing *dataset.txt file and seperate train and testing data)

# Evaluation result
Using `evaluation.py` get the output:
```
accuracy: 98.62%
```

# single_evaluation.py
script single_evaluation.py is for evaluation single person name get from the argument
```
$ python single_evaluation.py "xi jinping"
output:
probability russian names = 0.00%
probability chinese names = 78.57%
probability arabic names = 21.43%
$ python single_evaluation.py "vladimir putin"
output:
probability russian names = 94.90%
probability chinese names = 0.00%
probability arabic names = 5.10%
$ python single_evaluation.py "salman bin abdul-aziz al saud"
probability russian names = 0.00%
probability chinese names = 0.00%
probability arabic names = 100.00%
```
