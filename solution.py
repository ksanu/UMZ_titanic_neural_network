import numpy as np
import pandas
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
column_names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

train_X_in_tsv = pandas.read_csv('train/train.tsv', sep='\t', header=0, usecols=['Survived', 'Pclass',
                                                                            'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                                                                            'Fare', 'Cabin', 'Embarked'])

def data_prep(dataset):
    dataset['Sex'] = dataset['Sex'].replace('male', 1)
    dataset['Sex'] = dataset['Sex'].replace('female', 0)

data_prep(train_X_in_tsv)

ticket_col = train_X_in_tsv['Ticket']
ticket_type_list = list()
ticket_type_set = list()
p = re.compile('\s')
for row in ticket_col:
    list_of_words = p.split(row)
    if list_of_words.__len__() > 1:
        if ticket_type_set.__contains__(list_of_words[0]) == False:
            ticket_type_set.append(list_of_words[0])
        ticket_type_list.append(list_of_words[0])
    else:
        ticket_type_list.append("number")


ticket_type_col = pandas.DataFrame(data=ticket_type_list, columns=['Ticket type'])
train_X_in_tsv['Ticket'] = ticket_type_col['Ticket type']

train_y_in_tsv = train_X_in_tsv['Survived']
train_X_in_tsv.drop(columns=['Survived'], inplace=True)

test_X =  pandas.read_csv('test-A/in.tsv', sep='\t', header=None, names=["PassengerId", "Pclass", "Name",
                                                                        "Sex","Age", "SibSp", "Parch", "Ticket", "Fare",
                                                                        "Cabin", "Embarked"])
test_X.drop(columns=['PassengerId', 'Name'], inplace=True)

data_prep(test_X)

ticket_col = test_X['Ticket']
ticket_type_list = list()

p = re.compile('\s')
for row in ticket_col:
    list_of_words = p.split(row)
    if list_of_words.__len__() > 1:
        if ticket_type_set.__contains__(list_of_words[0]) == True:
            ticket_type_list.append(list_of_words[0])
        else:
            ticket_type_list.append("number")
    else:
        ticket_type_list.append("number")


ticket_type_col = pandas.DataFrame(data=ticket_type_list, columns=['Ticket type'])
test_X['Ticket'] = ticket_type_col['Ticket type']


def encoder(data):
    '''Map the categorical variables to numbers to work with scikit learn'''
    for col in data.columns:
        if data.dtypes[col] == "object":
            data[col].fillna('0', inplace=True)
            le = preprocessing.LabelEncoder()
            le.fit(data[col])
            data[col] = le.transform(data[col])
        else:
            data[col].fillna(0, inplace=True)

    return data

train_X_in_tsv = encoder(train_X_in_tsv)
test_X= encoder(test_X)

my_classifier = DecisionTreeClassifier()
my_classifier.fit(train_X_in_tsv, train_y_in_tsv)

y_out_predicted = my_classifier.predict(test_X)

with open('test-A/out.tsv', 'w') as output_file:
    for out in y_out_predicted:
        print('%d' % out, file=output_file)
