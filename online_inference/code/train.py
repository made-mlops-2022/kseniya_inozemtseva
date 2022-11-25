# Import packages
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import gzip


data = pd.read_csv('../data/heart_cleveland_upload.csv')


y = data.pop('condition')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


estimators = []
estimators.append(('logistic', LogisticRegression()))
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))


ensemble = VotingClassifier(estimators)


num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

pipe = Pipeline([
    ('scaler', ColumnTransformer([('scaler', StandardScaler(), num_cols)], remainder='passthrough')),
    ('model', ensemble)  # Ensemble Model
])


pipe.fit(X_train, y_train)


if __name__ == '__main__':
    print("Accuracy: %s%%" % str(round(pipe.score(X_test, y_test), 3) * 100))


joblib.dump(pipe, gzip.open('../model/model_binary.dat.gz', "wb"))

# pickle works differently when run from different locations
# hard to use with multiple launching use cases
#with open('../model/model.pkl', 'wb') as file:
#    pickle.dump(pipe, file)
