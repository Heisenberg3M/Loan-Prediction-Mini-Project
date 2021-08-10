# import the required libraries and then read the CSV file:
import pandas as pd
train = pd.read_csv('train_ctrUa4K.csv')
train.head()

'''We know that machine learning models take only numbers as inputs and can not process strings. 
So, we have to deal with the categories present in the dataset and convert them into numbers.'''
train['Gender']= train['Gender'].map({'Male':0, 'Female':1})
train['Married']= train['Married'].map({'No':0, 'Yes':1})
train['Loan_Status']= train['Loan_Status'].map({'N':0, 'Y':1})
# Here, we have converted the categories present in the Gender, Married and the Loan Status variable into numbers, simply using the map function of python.

# let’s check if there are any missing values in the dataset:
train.isnull().sum() # yes there are null values present

# we will remove all the rows which contain any missing values in them:
train = train.dropna()
train.isnull().sum() # null values are removed

# Next, we will separate the dependent (Loan_Status) and the independent variables:
X = train[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = train.Loan_Status
# X.shape, y.shape
'''For this particular project, I have only picked 5 variables that I think are most relevant.
 These are the Gender, Marital Status, ApplicantIncome, LoanAmount, and Credit_History and stored them in variable X. 
 Target variable is stored in another variable y. 
 And there are 480 observations available.'''

'''let’s move on to the model building stage.
 Here, we will first split our dataset into a training and validation set, so that we can train the model on the training
 set and evaluate its performance on the validation set.'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
'''We have split the data using the train_test_split function from the sklearn library keeping the test_size as 0.2 
 which means 20 percent of the total dataset will 
 be kept aside for the validation set.'''

# Next, we will train the random forest model using the training set:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4, random_state = 10)
model.fit(x_train, y_train)
'''Here, I have kept the max_depth as 4 for each of the trees of our random forest and stored the trained model 
in a variable named model.'''

# our model is trained, let’s check its performance on both the training and validation set:
from sklearn.metrics import accuracy_score
pred_test = model.predict(x_test)
accuracy_score(y_test, pred_test)
# 0.8020833333333334
''' The model is 80% accurate on the validation set.
 Let’s check the performance on the training set too:'''
pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)
# 0.8203125
# Performance on the training set is almost similar to that on the validation set. So, the model has generalized well.


# Finally, we will save this trained model so that it can be used in the future to make predictions on new observations:
import pickle
pickle_out = open("classifier.pkl", mode = "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
''' We are saving the model in pickle format and storing it as classifier.pkl. 
 This will store the trained model and we will use this while deploying the model.'''

'''
This completes the first five stages of the machine learning lifecycle. Next, we will explore the last stage which is model deployment. 
We will be deploying this loan prediction model so that it can be accessed by others.
And to do so, we will use Streamlit which is a recent and the simplest way of building web apps and deploying machine learning and deep learning models. 
'''