import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, accuracy_score, precision_score


# Assuming the TXT file is tab-separated
df = pd.read_csv('traffic.csv')
# Display the first few rows of the DataFrame (To test if its working)
print(df.head())

#potential pre-processing required


#Splitting the data into training and testing splits
X = df.drop("Total" , axis = 'columns').values #Drops target column using Total vehicles (FOR NOW)
y = df["Total"].values #retrieves the target column Total vehicles (FOR NOW)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.30, random_state = 42)
#Test_size is 30% and 70% is for training
