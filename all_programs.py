#program1

import numpy as np
import pandas as pd

class adaline:
    def __init__(self,learning_rate=0.01,n_iterations=100):
        self.learning_rate=learning_rate
        self.n_iterations= n_iterations
        self.weights = None
        
    def fit(self,X,y):
        self.weights=np.zeros(X.shape[1]+1);
        for i in range(self.n_iterations):
            net_input = np.dot(X,self.weights[1:])+ self.weights[0]
            predicted = net_input

            errors = y-predicted
            self.weights[1:]+= self.learning_rate*np.dot(X.T,errors)
            self.weights[0]+=self.learning_rate*np.sum(errors)
            
    def predict(self, X):
        net_input = np.dot(X,self.weights[1:])+ self.weights[0]`
        return net_input


data = pd.read_csv("data.csv")
X= data.iloc[:, :-1].values
y= data.iloc[:, -1].values
Adaline = adaline(learning_rate=0.01, n_iterations=10)
Adaline.fit(X,y)
print("weights: ", Adaline.weights)
prediction = Adaline.predict([[2,5]])
print("prediction is :" , prediction)


#program2
import csv
hypo = ['%','%','%','%','%','%']
data = []
with open("trainingdata.csv") as csv_file:
    readcsv = csv.reader(csv_file, delimiter =',')
    print("training example: ")
    for row in readcsv:
        print(row)
        if(row[len(row)-1]).upper()=='YES':
            data.append(row)
print("\npositive examples: ")
for i in data:
    print(i)
d = len(data[0])-1
hypo = list(data[0][:d])
print("\n find - s algorithm: ")
print("\n start:",hypo)

for i in range( len(data) ):
    for k in range( d):
        if hypo[k]!=data[i][k]:
            hypo[k]='?'
    print(f"after example {i+1}",hypo)

print("\nmost specific hypothesis:", hypo)


#program3
import csv

# Initialize hypotheses
# '%' = null (most specific), '?' = don't care (most general)
specific_h = ['%', '%', '%', '%', '%', '%']
general_h  = [['?','?','?','?','?','?'],
              ['?','?','?','?','?','?'],
              ['?','?','?','?','?','?'],
              ['?','?','?','?','?','?'],
              ['?','?','?','?','?','?'],
              ['?','?','?','?','?','?']]

data = []

# Step 1: Read CSV and collect all examples
with open('trainingdata.csv') as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')

    print("\nThe given training examples are:")
    for row in readcsv:
        print(row)
        data.append(row)

# Step 2: Initialize specific_h from the first positive example
for row in data:
    if row[-1].upper() == 'YES':
        d = len(row) - 1
        specific_h = list(row[:d])
        break

print("\nInitialization of specific_h:", specific_h)
print("Initialization of general_h:")
for g in general_h:
    print(g)

# Step 3: Process each training example
print("\nThe steps of Candidate Elimination algorithm are:")

for row in data:
    d = len(row) - 1

    if row[-1].upper() == 'YES':
        print("\nIf instance is Positive")
        for k in range(d):
            if row[k] != specific_h[k]:
                specific_h[k]   = '?'   # generalize S
                general_h[k][k] = '?'   # reset G at that position

    if row[-1].upper() == 'NO':
        print("\nIf instance is Negative")
        for k in range(d):
            if row[k] != specific_h[k]:
                general_h[k][k] = specific_h[k]  # specialize G
            else:
                general_h[k][k] = '?'             # keep G open

    print("specific_h:", specific_h)
    print("general_h :")
    for g in general_h:
        print(" ", g)

# Step 4: Remove fully general rows from G (all '?') — redundant
general_h = [g for g in general_h if g != ['?','?','?','?','?','?']]

# Step 5: Print final result
print("\nThe maximally specific hypothesis (S) is:")
print(specific_h)

print("\nThe general boundary hypothesis (G) is:")
for g in general_h:
    print(g)

#program5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('iris.csv')
X= data.iloc[:,:-1]
y = data.iloc[:,-1]
print("features:",list(X.columns))
print ("label:", y.name)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

classifier = GaussianNB()
classifier.fit(x_train,y_train)

print("model trained successfully!")
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_pred,y_test)

print('Predictions:', y_pred)
print('\nAccuracy: {:.2f}%'.format(accuracy * 100))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))


#program6
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score, classification_report

data = pd.read_csv('document.csv')
X= data['text']
y = data['label']
print("features:",X.name)
print ("label:", y.name)
tfidf=TfidfVectorizer()
X = tfidf.fit_transform(X)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


classifier = MultinomialNB()


# Train model
classifier.fit(x_train, y_train)

print("Model trained successfully!")


# Prediction
y_pred = classifier.predict(x_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)



print("Predictions:", y_pred)

print('\nAccuracy: {:.2f}%'.format(accuracy * 100))

print('\nClassification Report:')

print(classification_report(y_test, y_pred))

#program7
# P7 — Bayesian Network (Heart Disease)
**Dataset:** heart.csv (Cleveland Heart Disease)
**Model:** BayesianModel + MaximumLikelihoodEstimator
**Task:** Infer probability of heart disease given evidence
# ─── STEP 1: IMPORT LIBRARIES ───────────────────────────────
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ─── STEP 2: LOAD DATASET ───────────────────────────────────
data = pd.read_csv('heart.csv')       # ← change filename here
data = data.replace('?', np.nan)      # replace missing values

print('Shape:', data.shape)
print('\nFirst 5 rows:')
print(data.head())
print('\nColumn names:', list(data.columns))

# ─── STEP 3: SPLIT FEATURES AND LABEL ───────────────────────
# No X/y split needed here — Bayesian Network uses ALL columns
# The network itself defines which columns depend on which
X = data.iloc[:, :-1]   # features (for reference)
y = data.iloc[:,  -1]   # target: heartdisease column

print('Features:', list(X.columns))
print('Target  :', y.name)
print('\nTarget value counts:')
print(y.value_counts())

# ─── STEP 4: TRAIN/TEST SPLIT ───────────────────────────────
# Note: Bayesian Networks don't split data the usual way.
# The ENTIRE dataset is used to LEARN the probability tables.
# Inference (querying) acts as the 'prediction' step.
print('Total samples used for learning:', len(data))
print('(All data used to learn CPDs — no train/test split needed)')

# ─── STEP 5: CREATE AND TRAIN MODEL ─────────────────────────
# Define the DAG (Directed Acyclic Graph) — which node affects which
# Format: ('cause', 'effect')
model = BayesianModel([
    ('age',          'trestbps'),
    ('age',          'fbs'),
    ('sex',          'trestbps'),
    ('exang',        'trestbps'),
    ('trestbps',     'heartdisease'),
    ('fbs',          'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'),
    ('heartdisease', 'chol')
])

# Learn Conditional Probability Distributions (CPDs) from data
# MaximumLikelihoodEstimator = counts frequencies from data
print('Learning CPDs using Maximum Likelihood Estimator...')
model.fit(data, estimator=MaximumLikelihoodEstimator)
print('Model trained successfully!')

# ─── STEP 6: PREDICT AND EVALUATE ───────────────────────────
# VariableElimination = inference engine to answer queries
# It computes P(target | evidence) using the learned CPDs
infer = VariableElimination(model)

print('Inferencing with Bayesian Network:')
print('=' * 45)

# Query 1: What is P(heartdisease | age=28)?
print('\n1. Probability of HeartDisease given Age = 28')
q1 = infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q1)

# Query 2: What is P(heartdisease | chol=100)?
print('\n2. Probability of HeartDisease given Cholesterol = 100')
q2 = infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q2)

# ─── STEP 7: PREDICT ON NEW SAMPLE ─────────────────────────
# Ask: what is the probability of heartdisease for a new patient?
print('3. Probability of HeartDisease given Age=50 and Sex=1 (male)')
q3 = infer.query(
    variables=['heartdisease'],
    evidence={'age': 50, 'sex': 1}
)
print(q3)

print('\n4. Probability of HeartDisease given fbs=1 (high blood sugar)')
q4 = infer.query(
    variables=['heartdisease'],
    evidence={'fbs': 1}
)
print(q4)
 #----------#

