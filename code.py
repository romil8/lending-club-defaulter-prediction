# --------------
import pandas as pd
from sklearn.model_selection import train_test_split as tts

# Code starts here
df= pd.read_csv(filepath_or_buffer=path, compression='zip', low_memory = False)
df.head()
X= df.drop(['loan_status'],1)
y= df['loan_status']

X_train,X_test,y_train,y_test= tts(X,y,test_size= 0.25, random_state= 4)
# Code ends here


# --------------
# Code starts  here
col= df.isnull().sum()
print(col)

print('-'*20)

col_drop= col[col>0.25*len(df)].index.tolist()
print(col_drop)

for x in df:
    if df[x].nunique() == 1 :
        col_drop.append(x)

X_train= X_train.drop(col_drop,1)
X_test= X_test.drop(col_drop,1)
# Code ends here


# --------------
import numpy as np


# Code starts here
y_train= np.where((y_train == 'Fully Paid') |(y_train == 'Current'), 0, 1)
y_test= np.where((y_test == 'Fully Paid') |(y_test == 'Current'), 0, 1)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder


# categorical and numerical variables
cat = X_train.select_dtypes(include = 'O').columns.tolist()
num = X_train.select_dtypes(exclude = 'O').columns.tolist()

# Code starts here
for x in cat:
    mode = X_train[x].mode()[0]
    X_train[x].fillna(mode, inplace = True)

for x in num:
    mean = X_train[x].mean()
    X_train[x].fillna(mean,inplace = True)

# Test Data
for x in cat:
    mode = X_train[x].mode()[0]
    X_test[x].fillna(mode,inplace = True)

for x in num:
    mean = X_train[x].mean()
    X_test[x].fillna(mean,inplace = True)


# Label encoding

le = LabelEncoder()
for x in cat:
    
    X_train[x] = le.fit_transform(X_train[x])
    X_test[x] = le.fit_transform(X_test[x])


# Code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

# Code starts here
rf = RandomForestClassifier(random_state= 42,max_depth=2,min_samples_leaf=5000)

rf.fit(X_train,y_train)

accuracy = rf.score(X_test,y_test)

y_pred = rf.predict(X_test)

# Store the different evaluation values.

f1 = f1_score(y_test, rf.predict(X_test))
precision = precision_score(y_test, rf.predict(X_test))
recall = recall_score(y_test, rf.predict(X_test))
roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



# Plot the auc-roc curve

score = roc_auc_score(y_pred , y_test)
y_pred_proba = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Random Forrest, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()



# Code ends here


# --------------
from xgboost import XGBClassifier

# Instantiate the  XGBoost model
xgb = XGBClassifier(learning_rate=0.0001)

# Fit the model on train

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)

# Store the different evaluation values.

f1 = f1_score(y_test, xgb.predict(X_test))
precison = precision_score(y_test, xgb.predict(X_test))
recall = recall_score(y_test, xgb.predict(X_test))
roc_auc = roc_auc_score(y_test, xgb.predict(X_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Plot the auc-roc curve

score = roc_auc_score(y_test,y_pred)
y_pred_proba = xgb.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="XG Boost, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


