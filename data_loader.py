import pandas as pd
import numpy as np

def data_loader():
    data1 = pd.read_csv("data/diabetes.csv")
    # diabetes_data_copy = data1.copy(deep = True)
    # diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    # diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
    # diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
    # diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
    # diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
    # diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

    data2 = pd.read_csv('data/ionosphere.data', header=None)
    data2.drop(columns=[1], inplace=True)
    data2['label'] = data2[34] == 'g'

    data3 = pd.read_csv('data/winequality-red.csv', sep=";")
    data3['label'] = data3['quality'] > 5
    data3['label'] = data3['label'].astype(int)

    data4 = pd.read_csv('data/winequality-white.csv', sep=";")
    data4['label'] = data4['quality'] > 6
    data4['label'] = data4['label'].astype(int)

    X1, y1 = data1[["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]], data1[['Outcome']]
    X2, y2 = data2.iloc[:, :34], data2['label']
    X3, y3 = data3[data3.columns[:-2]], data3['label']
    X4, y4 = data4[data4.columns[:-2]], data4['label']

    return {
        'diabetes': (X1, y1),
        'ionosphere': (X2, y2),
        'red-wine': (X3, y3),
        'white-wine': (X4, y4)
    }