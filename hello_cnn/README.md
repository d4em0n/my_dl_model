# Hello CNN!
This is my first experiment with deep learning. this CNN model classified country based on person name, this CNN trained using chinese and russian names and can classified whether a person's name is chinese or russian name.

# Datasets
- There are 9800 chinese name and 9800 russian name (total 19600 person's name)
- This model trained with 13720 random mixed chinese and russian name the rest of 5880 is used for evaluation/testing data.

# Evaluation result
Using `evaluation.py` get the output:
```
accuracy: 99.81%
```
which is nice :).

# single_evaluation.py
script single_evaluation.py is for evaluation single person name get from the argument
```
$ python single_evaluation "xi jinping"
output: This is chinese name
$ python single_evaluation "vladimir putin"
output: This is russian name
```
