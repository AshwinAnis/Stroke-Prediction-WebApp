# Import Packages
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Train Test Split
from sklearn.model_selection import train_test_split

# Scaler

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder


# Pipeline

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer


# Models

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Validation

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# Import Data
stroke_data = pd.read_csv('stroke.csv')

# Removing Null Values
sd = stroke_data.dropna()


sd = sd.reset_index()
y = sd['stroke']
X = sd[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']]
# X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

# Splitting Columns into Countinuous and Caetegorical 

cat_cols = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']
num_cols = ['age','avg_glucose_level','bmi']


numeric_transformer =  Pipeline(steps=[('scaler', RobustScaler())])
categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols),('cat', categorical_transformer, cat_cols)], remainder='drop', n_jobs=1)

clf = Pipeline(steps=[('preprocessor', preprocessor),
 ('classifier', LogisticRegression(random_state=100))])


clf.fit(X, y)

#Save model
pickle.dump(clf, open('model.pkl','wb'))

# Loading Model
model  = pickle.load(open('model.pkl','rb'))


