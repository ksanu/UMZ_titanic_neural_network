import numpy as np
import pandas
import re
from sklearn.tree import DecisionTreeClassifier

column_names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

train_X_in_tsv = pandas.read_csv('train/train.tsv', sep='\t', header=0, usecols=['Survived', 'PassengerId', 'Pclass',
                                                                            'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                                                                            'Fare', 'Cabin', 'Embarked'])

def data_prep(dataset):
    dataset['Sex'] = dataset['Sex'].replace('male', 1)
    dataset['Sex'] = dataset['Sex'].replace('female', 0)

data_prep(train_X_in_tsv)

ticket_col = train_X_in_tsv['Ticket']
ticket_type_list = list()
p = re.compile('\s')
for row in ticket_col:
    list_of_words = p.split(row)
    if list_of_words.__len__() > 1:
        if ticket_type_list.__contains__(list_of_words[0]) == False:
            ticket_type_list.append(list_of_words[0])


print (ticket_type_list)



print(train_X_in_tsv)
