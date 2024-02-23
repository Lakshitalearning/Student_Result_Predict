
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression #LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('stuperformance.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scaling independent variables
sc_X = StandardScaler()
X_train[:, 2:] = sc_X.fit_transform(X_train[:, 2:])
X_test[:, 2:] = sc_X.transform(X_test[:, 2:])  # Use the same scaling for testing
#scaling dependent variable
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))  # Reshape to 2D for scaling
y_test = sc_y.transform(y_test.values.reshape(-1, 1))

models={'Linear Regression':LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Support Vector Machine': SVR(kernel='rbf'),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor()
}
model_selector=st.selectbox("Select any model:",list(models.keys()))

regressor = models[model_selector]
regressor.fit(X_train, y_train)

score=regressor.score(X_test,y_test)
st.write(f'{model_selector} Model Score : {score}')

X_df=pd.DataFrame(X)
st.header('STUDENT PERFORMANCE PREDICTOR')
st.sidebar.header('USER INPUT')  ##### I DONT NEED SIDEBAR I WANT IT AT BOTTOM
user_inputs={}
for column in X_df.columns:
    user_inputs[column]=st.sidebar.number_input(f'Enter {column}',value =X[column].mean())

# input_ = np.array([[1,0,8,45,4,6]])
# input_[:, 2:] = sc_X.transform(input_[:, 2:])
# y_1pred = regressor.predict(input_)
# y_1pred_unscaled = sc_y.inverse_transform(y_1pred.reshape(-1,1))
# print(y_1pred_unscaled)
# user_inputs_df=pd.DataFrame([user_inputs])
# user_inputs_df.iloc[:, 2:] = sc_X.transform(user_inputs_df.iloc[:, 2:])
# prediction=regressor.predict(user_inputs_df)
# prediction_unscaled=sc_y.inverse_transform(prediction.reshape(-1,1))
# st.write(f'Predicted Student Performance : {prediction_unscaled}')
user_inputs=sc_X.transform(user_inputs)
prediction=regressor.predict(pd.DataFrame([user_inputs]))
prediction_unscaled=sc_y.inverse_transform(prediction.reshape(-1,1))
st.write(f'Predicted Student Performance : {prediction[0]}')
