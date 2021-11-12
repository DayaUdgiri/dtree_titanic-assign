# necessary Imports
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier

# loading the data
url = "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(url)

# fetching features(X) and label(Y)
x = titanic[['Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare']]

y = titanic['Survived']

# here Sex column is Categorical as it has text data - 'male' and 'female'
# we can convert the categorical into numrical by using pandas get_dummies method
dummies=pd.get_dummies(x.Sex)

#now lets concatenate these dummies columns into our original dataset x
x=pd.concat([x,dummies],axis='columns')

x=x.drop(columns='Sex')
meanofage=x.Age.mean()
x.Age.fillna(meanofage,inplace=True)

# As the range of values of our featureset varies, let's perform standard scaling of it
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

# splitting the dataset into Train and Test sets
xsc_train,xsc_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.33,random_state=43)

dtree = DecisionTreeClassifier()

dtree.fit(xsc_train,y_train)

pruning_path = dtree.cost_complexity_pruning_path(xsc_train,y_train)

ccp_alpha_vals = pruning_path.ccp_alphas

d_trees=[]
for ccp in ccp_alpha_vals:
    dt_m = DecisionTreeClassifier(ccp_alpha=ccp)
    dt_m.fit(xsc_train, y_train)
    d_trees.append(dt_m)

d_tree_opti = DecisionTreeClassifier(ccp_alpha=0.0052)

d_tree_opti.fit(xsc_train,y_train)

print("Training Score of optimal Decision Tree is = " ,d_tree_opti.score(xsc_train,y_train))
print("Testing Score of optimal Decision Tree is = ", d_tree_opti.score(xsc_test,y_test))

filename = 'Decision_Tree_opti_model.pickle'

pickle.dump(d_tree_opti,open('Decision_Tree_opti_model.pickle','wb'))

loaded_dt_model=pickle.load(open('Decision_Tree_opti_model.pickle','rb'))

print("predicted survival= ", loaded_dt_model.predict(scaler.transform([x.iloc[0]]))," wheras expected Survival= ",y[0])
print("predicted survival= ", loaded_dt_model.predict(scaler.transform([x.iloc[2]]))," wheras expected Survival= ",y[2])
print("predicted survival= ", loaded_dt_model.predict(scaler.transform([x.iloc[4]]))," wheras expected Survival= ",y[4])
print("predicted survival= ", loaded_dt_model.predict(scaler.transform([x.iloc[5]]))," wheras expected Survival= ",y[5])