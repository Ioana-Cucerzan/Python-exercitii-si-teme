import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("=" * 70)
print("MODULE 04: EXERCITII PREPROCESARE & PIPELINES")
print("=" * 70)

# ============================================================================
# EXERCITIU 1: Identifica și Corectează Data Leakage
# ============================================================================
print("\n[EXERCITIU 1] Identifica și Corectează Data Leakage")
print("-" * 70)
"""
print("""
#ENUNȚ:
#Următorul cod conține o GREȘEALĂ de data leakage.
#Identifică greșeala și rescrie codul corect!
      
#     # ❌ COD CU LEAKAGE:

# Scenariul GREȘIT
X = pd.read_csv('date.csv')              # Încărcăm datele într-un DataFrame 
y = X['target']
X = X.drop('target', axis=1)

# Scalerul învață din TOATE datele!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apoi splituiesc
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Entreneaza modelul
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

"""
SARCINI:
1. Explică ce e greșit în codul de mai sus
2. Rescrie codul CORECT
3. Explică de ce e important să evit data leakage

# RĂSPUNS:
1) Greșeala în codul de mai sus este că scalerul (StandardScaler) este antrenat pe întregul set de date (X), inclusiv pe datele de testare. 
   Acest lucru duce la data leakage, deoarece informațiile din setul de testare sunt folosite pentru a transforma datele, ceea ce poate duce la o performanță artificial 
   crescută a modelului.

2) Codul corect ar trebui să fie:
"""
# Scenariul CORECT
X = pd.read_csv('date.csv') 
y = X['target']
X = X.drop('target', axis=1)
# Apoi splituiesc
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Scalerul învață doar din datele de antrenament!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform și datele de testare folosind același scaler
X_test_scaled = scaler.transform(X_test)    
# Entreneaza modelul
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


"""
3) Este important să evit data leakage pentru că poate duce la o evaluare nerealistă a performanței modelului. Dacă modelul are acces la informații din setul de testare în timpul antrenamentului, 
   acesta poate învăța să recunoască aceste informații și să obțină rezultate foarte bune pe setul de testare, dar va performa slab pe date noi, neîntâlnite anterior. 
   Evitarea data leakage-ului asigură că modelul este evaluat corect și că performanța sa reflectă capacitatea sa de a generaliza la date noi."""