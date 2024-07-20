import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("diabetes.csv")

x = data.iloc[:, 0:8]
y = data.iloc[:, -1]
X = x.to_numpy()
Y = y.to_numpy().reshape(-1, 1)

patient = data[data.Outcome == 1]
healthy = data[data.Outcome == 0]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train.reshape(y_train.shape[0]))

pre = []
t = 0
wrong = 0
for i in X_test:
    pre.append([knn.predict(i.reshape(1, -1))[0], Y[t, 0]])
    if knn.predict(i.reshape(1, -1))[0] != Y[t, 0]:
        wrong += 1
    t += 1

result = pd.DataFrame(data=pre, columns=['prediction', 'real'])

scorelist = []
for i in range(1, 30):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train.reshape(y_train.shape[0]))
    scorelist.append([i,knn2.score(X_test, y_test)])

scorePD = pd.DataFrame(data=scorelist, columns=['n_neighbour', 'score'])

max_score = scorePD.loc[scorePD['score'].idxmax()]

while True:
    print("type 1 to show the graph of patient and healthy people")
    print("type 2 to see the best n neighbour for this specific job")
    print("type 3 scatter the graph of n neighbour and score")
    print("type 4 to predict")
    opt = int(input("Type a number : "))

    if opt == 1:
        plt.scatter(healthy.Age, healthy.Glucose, color='green', label='healthy', alpha=0.4)
        plt.scatter(patient.Age, patient.Glucose, color='red', label='patient', alpha=0.4)
        plt.xlabel('age')
        plt.ylabel('glucose')
        plt.legend()
        plt.show()

    elif opt == 2:
        print("**************")
        print(max_score)
        print("**************")

    elif opt == 3:
        plt.scatter(scorePD['n_neighbour'], scorePD['score'], color='blue')
        plt.xlabel('n_neighbour')
        plt.ylabel('score')
        plt.show()

    elif opt == 4:
        pregnancy = float(input("Enter pregnancy : "))
        glucose = float(input("Enter glucose : "))
        blood_pressure = float(input("Enter blood pressure : "))
        skin_tickness = float(input("Enter skin tickness : "))
        insulin = float(input("Enter insulin : "))
        bmi = float(input("Enter bmi : "))
        DiabetesPedigreeFunction = float(input("Enter Diabetes Pedigree Function : "))
        age = float(input("Enter Age : "))
        x_pred = np.array([[pregnancy, glucose, blood_pressure, skin_tickness, insulin, bmi, DiabetesPedigreeFunction, age]])
        X_pred = sc.transform(x_pred)
        print("prediction : " + str(knn.predict(X_pred)[0]))
        

    print("###################################")





