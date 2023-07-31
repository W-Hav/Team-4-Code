#This is designed to run in kaggle.com.

import pandas as pd
import numpy as np

df1 = pd.read_csv("/kaggle/input/disease-prediction-using-machine-learning/Training.csv")
df = df1.drop("fluid_overload", axis = "columns")

df = df.set_axis(['Do you have an itch', 'Do you have a skin rash', 'Do you have nodal skin eruptions', 'Are you continuously sneezing', 'Have you been shivering', 'Have you experienced chills', 'Do you have joint pain', 'Do you have stomach pain', 'Do you have acid reflux', 'Do you have tongue ulcers', 'Have you experienced muscle loss', 'Have you recently been vomiting', 'Do you experience pain when you urinate', 'Is there blood in your urine', 'Do you feel fatigued', 'Have you recently gained weight', 'Do you have anxiety', 'Are your hands and feet cold', 'Have you been experiencing mood swings', 'Have you recently lost weight', 'Have you been feeling restless', 'Have you been feeling lethargic', 'Have you experienced throat patches', 'Do you have irregular sugar levels', 'Do you have a cough', 'Have you recently had a high fever', 'Do you have sunken eyes', 'Have you been feeling breathless', 'Have you been sweating uncontrollably', 'Do you feel dehydrated', 'Have you experienced indigestion', 'Have you recently experienced a headache', 'Do you have yellowish skin', 'Do you have dark urine', 'Do you feel nauseous', 'Have you felt a recent loss of appetite', 'Do you have pain behind your eyes', 'Do you have back pain', 'Have you experienced constipation', 'Do you have abdominal pain', 'Have you recently had diarrhea', 'Have you recently experienced a mild fever', 'Is your urine yellow', 'Are your eyes yellow', 'Do you have acute liver failure', 'Is your stomach swelling', 'Do you have swollen lymph nodes', 'Are you generally feeling unwell', 'Have you recently experienced blurred and distorted vision', 'Are you coughing up phlegm', 'Is your throat irritated', 'Are your eyes red', 'Do you feel pressure in your sinuses', 'Do you have a runny nose', 'Are you congested', 'Do you have chest pain', 'Have you experienced weakness in your limbs', 'Is your heart rate unusually fast', 'Do you experience pain during bowel movements', 'Do you feel pain in your anal region', 'Do you have bloody stool', 'Do you feel irritation in your anus', 'Do you have neck pain', 'Have you experienced dizziness', 'Have you experienced cramps', 'Have you experienced bruising', 'Are you obese', 'Are your legs swollen', 'Do you have any swollen blood vessels', 'Are your eyes and face puffy', 'Do you have an enlarged thyroid', 'Are your nails brittle', 'Do you have any swollen extremities', 'Have you experienced excessive feelings of hunger', 'Do you have any extramarital relations', 'Do you have dry and tingling lips', 'Have you noticed your speech being slurred', 'Do you have knee pain', 'Do you have hip pain', 'Do you have muscle weakness', 'Do you have a stiff neck', 'Do you have any swollen joints', 'Are your movements stiff', 'When you move, do you feel as if everything is spinning', 'Have you lost your balance more than usual', 'Have you experienced feelings of unsteadiness', 'Do you feel that one side of your body is weak', 'Are you experiencing a loss of smell', 'Are you experiencing bladder discomfort', 'Does your urine smell foul', 'Do you continually feel the need to urinate', 'Have you experienced a significant passage of gasses', 'Have you experienced internal itching', 'Do you appear visually ill', 'Do you feel depressed', 'Are you irritable', 'Have you experienced muscle pain', 'Are you having trouble thinking clearly', 'Do you have red spots over your body', 'Have you experienced belly pain', 'Are you experiencing abnormal menstruation', 'Do you have patchy or irregular skin discoloration', 'Are your eyes watering', 'Have you noticed an increase in your appetite', 'Do you produce unusually large amounts of urine', 'Do you have a family history of the condition you are concerned about', 'Do you have mucus in your sputum', 'Is your sputum "rusty"', 'Do you lack concentration', 'Have you noticed any visual distubances', 'Have you recently received a blood transfusion', 'Have you recently received any unsterile injections', 'Have you recently recovered from a coma', 'Have you experienced stomach bleeding', 'Has your abdomen become notably swollen', 'Do you have a history of alcohol consumption', 'Have you recently consumed a significant amount of fluids', 'Do you have blood in your sputum', 'Are the veins on your calf prominently showing', 'Have you been feeling palpitations', 'Does it hurt to walk', 'Do you have pus filled pimples', 'Do you have blackheads', 'Have you experienced scarring', 'Is your skin peeling', 'Have you been exposed to large amounts of silver', 'Do you have small dents in your nails', 'Do you have inflamed nails', 'Do you have blisters', 'Do you have a red sore around your nose', 'Do you have a yellow scar', 'prognosis', 'Unnamed: 133'], axis='columns', copy=False)
df

x1 = df.drop("prognosis", axis = "columns")

x = x1.drop("Unnamed: 133", axis = "columns")

y = df["prognosis"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

neighbors = np.arange(1, 176)

testAcc = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    print(i)
    print(k)

test_acc = np.empty(len(neighbors))

from sklearn.neighbors import KNeighborsClassifier
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    
    knn.fit(x_train, y_train)
    
    test_acc[i] = knn.score(x_test, y_test)

c = x.columns

symptoms = np.zeros(len(c))

for x in range(len(c)):
    l = input(df.columns[x] +"? Y or N?")
    if (l=="Y" or l=="y"):
        p=1
        symptoms[x]=p
    elif (l=="N" or l == "n"):
        p=0
        symptoms[x]=p
    else:
        while True: 
            print("No valid input received. Please try again.")
            l = input(df.columns[x] +"? Y or N?") 
            if (l=="Y" or l=="y"):
                p=1
                symptoms[x]=p
            elif (l=="N" or l == "n"):
                p=0
                symptoms[x]=p
                break
        
symptoms
symptoms.reshape(1, -1)
new_output = knn.predict([symptoms])
new_output
