# Story Understanding


This project is about story understanding. The system uses four-sentence stories with two possible ending sentences. Then the system predicts the right ending sentence. Eventually, the system's performance of story understanding is evaluated.


## Getting started


Clone the repository. Make sure to keep the same structure of the files and folders. Thus the preprocessing file is not in a folder. Furthermore, the two folders Data and Systems contain multiple files. 


## Prerequisites


* Python 3.6.7
* Pandas 

```
pip3 install pandas==0.24.2
```
* Scikit-learn
```
pip3 install -U scikit-learn==0.20.2
```
* NLTK
```
pip3 install -U nltk==3.4
```
* VADER 
```
pip3 install vaderSentiment==3.2.1
```


## Installing


The train and development data are already included in the folder Data. If you want to create your own train and development set you can use the following command:
```
python3 preprocessing.py
```


## Running systems

First go to the folder Systems.
To run the first-sentence system, use the following command:
```
python3 first-sentence_system.py
```
To run the fourth-sentence system, use the following command:
```
python3 fourth-sentence_system.py
```


## Author


Lisa Warrink - lisa3217124


## Notes


The systems run on the train and development or test set. The train and development or test set should be from the same corpus, namely the 2016 or 2018 corpus. Here is an example for using the 2016 corpus:
```
df_train = pd.read_csv('../Data/train2016.csv', delimiter=',')
df_dev = pd.read_csv('../Data/dev2016.csv', delimiter=",") #or
df_test = pd.read_csv("../Data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv", delimiter=",")
```
Another example for using the 2018 corpus:
```
df_train = pd.read_csv('../Data/train2018.csv', delimiter=',')
df_dev = pd.read_csv('../Data/dev2018.csv', delimiter=",") #or
df_test = pd.read_csv("../Data/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv", delimiter=",")
```

The default for both systems is the 2016 corpus. The corpora used, are the Story Cloze Test Spring 2016 and Story cloze Test Winter 2018. These corpora are available at https://competitions.codalab.org/competitions/15333#learn_the_details-get-data.

Furthermore, if you only want to use a few features for the system, you should specify these for the training part as well as developing or testing part.
