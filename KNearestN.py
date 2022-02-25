import pandas as pd
import  numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

data=pd.read_csv('KNN_Dataset.csv',sep=',')
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
# change value of 0 to NaN
for column in zero_not_accepted:
    data[column] = data[column].replace(0,np.NaN)
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.NaN,mean)

x = data.iloc[:,0:8]
y = data.iloc[:,8]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

model=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
model.fit(x_train,y_train)
# score=model.score(x_test,y_test)
pred=model.predict(x_test)
print(pred)
ev=confusion_matrix(y_test,pred)
print(ev)
sc=f1_score(y_test,pred)
print(sc)
acc=accuracy_score(y_test,pred)
print(acc)

