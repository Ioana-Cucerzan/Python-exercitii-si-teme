"""
Module 02: Pandas - Exerciții
=============================

6 exerciții progresive cu scenarii realiste de data cleaning și feature engineering.

Instrucțiuni:
1. Cititi exercițiul
2. Scrieți codul în secțiunea de soluție
3. Comparați cu SOLUȚIE (desfășurate în jos)

Run: python exercitii.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("EXERCIȚII PANDAS - PROGRESIVE")
print("="*80)


# ============================================================================
# EXERCIȚIU 1: Data Cleaning - Handling Missing Values
# ============================================================================

print("\n" + "="*80)
print("EXERCIȚIU 1: Data Cleaning - Valori Lipsă")
print("="*80)

print("""
CONTEXT:
Aveți dataframe-ul 'df_sales' cu valori lipsă în coloanele 'categorie' și 'pret_unitar'.

SARCINI:
a) Afișați câte valori lipsă are fiecare coloană
b) Completați 'categorie' cu valoarea cea mai frecventă (modus)
c) Completați 'pret_unitar' cu mediana
d) Verificați că nu mai sunt valori lipsă

HINT: isna(), sum(), mode(), median(), fillna()
""")

print("\n--- SOLUȚIE ---")

# ============================================================================
# EXERCIȚIU 1: Data Cleaning - Handling Missing Values
# ============================================================================

print("\n" + "="*80)
print("EXERCIȚIU 1: Data Cleaning - Valori Lipsă")
print("="*80)

print("""
CONTEXT:
Aveți dataframe-ul 'df_sales' cu valori lipsă în coloanele 'categorie' și 'pret_unitar'.

SARCINI:
a) Afișați câte valori lipsă are fiecare coloană
b) Completați 'categorie' cu valoarea cea mai frecventă (modus)
c) Completați 'pret_unitar' cu mediana
d) Verificați că nu mai sunt valori lipsă

HINT: isna(), sum(), mode(), median(), fillna()
""")

print("\n--- SOLUȚIE ---")
                         
# Dataset 1: Vânzări de produse
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
sales_data = {
    'vanzare_id': range(1, 101),
    'data': np.random.choice(dates, 100),
    'produs': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'], 100),
    'categorie': np.random.choice(['Electronics', 'Accessories'], 100),
    'cantitate': np.random.randint(1, 20, 100),
    'pret_unitar': np.random.uniform(50, 2000, 100),
    'vanzator': np.random.choice(['Ana', 'Bob', 'Carol', 'David'], 100),
}
df_sales = pd.DataFrame(sales_data)

# Adăugare valori lipsă
df_sales.loc[np.random.choice(df_sales.index, 5, replace=False), 'categorie'] = np.nan
df_sales.loc[np.random.choice(df_sales.index, 3, replace=False), 'pret_unitar'] = np.nan

# Dataset 2: Studenți și note
students_data = {
    'student_id': range(1001, 1031),
    'nume': np.random.choice(['Ion', 'Andrei', 'Mircea', 'Vlad', 'Cristian',
                               'Elena', 'Sofia', 'Dana', 'Ioana', 'Alexandra'], 30),
    'curs': np.random.choice(['Python', 'SQL', 'ML', 'Web'], 30),
    'nota_test1': np.random.randint(40, 100, 30),
    'nota_test2': np.random.randint(40, 100, 30),
    'absente': np.random.randint(0, 15, 30),
}
df_students = pd.DataFrame(students_data)
df_students.loc[np.random.choice(df_students.index, 2, replace=False), 'nota_test2'] = np.nan

# Dataset 3: Departamente și angajați
departments = pd.DataFrame({
    'depart_id': [1, 2, 3, 4],
    'depart_name': ['IT', 'HR', 'Finance', 'Marketing'],
    'buget': [100000, 50000, 80000, 60000]
})

employees = pd.DataFrame({
    'emp_id': range(1001, 1021),
    'nume': np.random.choice(['Ana', 'Bob', 'Carol', 'David', 'Elena'], 20),
    'depart_id': np.random.choice([1, 2, 3, 4], 20),
    'salariu': np.random.uniform(2000, 6000, 20),
    'ani_exp': np.random.randint(0, 20, 20)
})

print("\nDataset-uri create pentru exerciții.")
print(f"Sales: {df_sales.shape[0]} rânduri")
print(f"Students: {df_students.shape[0]} rânduri")
print(f"Employees: {employees.shape[0]} rânduri")
                   


#a) Afișați câte valori lipsă are fiecare coloană
print("\na) Valori lipsă per coloană:") 
print(df_sales.isna().sum())    

#b) Completați 'categorie' cu valoarea cea mai frecventă (modus)
modus_categorie = df_sales['categorie'].mode()[0]
df_sales['categorie'] = df_sales['categorie'].fillna(modus_categorie)

#c) Completați 'pret_unitar' cu mediana
mediana_pret = df_sales['pret_unitar'].median()     
df_sales['pret_unitar'] = df_sales['pret_unitar'].fillna(mediana_pret)

#d) Verificați că nu mai sunt valori lipsă
print("\nd) Verificare valori lipsă după completare:")
print(df_sales.isna().sum())




print("\n" + "="*80)
print("EXERCIȚIU 2: Feature Engineering - Creare Coloane Noi")
print("="*80)

print("""
CONTEXT:
Pe df_sales, trebuie să calculați informații utile.

SARCINI:
a) Creați coloană 'valoare_totala' = cantitate * pret_unitar
b) Creați coloană 'pret_range' care să clasifice în: 'Cheap' (<300), 'Medium' (300-1000), 'Expensive' (>1000)
c) Creați coloană 'zi_saptamani' din coloana 'data' (0=Monday, 6=Sunday)
d) Afișați statistici pe 'pret_range'

HINT: apply(), dt.dayofweek, value_counts()
""")

print("\n--- SOLUȚIE ---")

# a) Creați coloană 'valoare_totala' = cantitate * pret_unitar
df_sales['valoare_totala'] = df_sales['cantitate'] * df_sales['pret_unitar']
# b) Creați coloană 'pret_range' care să clasifice în: 'Cheap' (<300), 'Medium' (300-1000), 'Expensive' (>1000)
def classify_price(price):
    if price < 300:
        return 'Cheap'
    elif price <= 1000:
        return 'Medium'
    else:
        return 'Expensive'
df_sales['pret_range'] = df_sales['pret_unitar'].apply(classify_price)
# c) Creați coloană 'zi_saptamani' din coloana 'data' (0=Monday, 6=Sunday)
df_sales['zi_saptamani'] = df_sales['data'].dt.dayofweek
# d) Afișați statistici pe 'pret_range'
print("\nd) Statistici pe 'pret_range':")
print(df_sales['pret_range'].value_counts())


        