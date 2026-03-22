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

# ============================================================================
# EXERCIȚIU 3: Groupby și Agregare - Analiza Vânzărilor
# ============================================================================

print("\n" + "="*80)
print("EXERCIȚIU 3: Groupby & Agregare - Analytics")
print("="*80)

print("""
CONTEXT:
Trebuie să analizați vânzările pe diferite dimensiuni.

SARCINI:
a) Câte vânzări a făcut fiecare vânzător (sort descrescător)
b) Valoare totală vândută per produs (sort descrescător)
c) Pentru fiecare produs: min, max, media prețului
d) Cât de mulți clienți per zi_saptamana

HINT: groupby(), size(), agg(), sort_values()
""")

print("\n--- SOLUȚIE ---")
import pandas as pd
df_sales = pd.DataFrame({
    'vanzator': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'],
    'produs': ['Laptop', 'Telefon', 'Laptop', 'Tableta', 'Telefon', 'Tableta'],
    'pret_unitar': [1000, 500, 950, 300, 450, 320],
    'cantitate': [1, 2, 1, 3, 1, 2],
    'valoare_totala': [1000, 1000, 950, 900, 450, 640],
    'zi_saptamana': [0, 1, 0, 2, 1, 2]  # Luni=0, Marți=1, Miercuri=2
})


import pandas as pd


# DATAFRAME Exercitiu 4


df_sales = pd.DataFrame({
    "vanzator": ["Ana", "Mihai", "Ioana", "Ana", "Mihai", "Ioana"],
    "produs": ["Laptop", "Monitor", "Laptop", "Mouse", "Monitor", "Laptop"],
    "categorie": ["Electronics", "Electronics", "Electronics", "Electronics", "Electronics", "Electronics"],
    "pret_unitar": [3500, 900, 3500, 80, 900, 3500],
    "valoare_totala": [3500, 900, 3500, 80, 900, 3500],
    "zi_saptamana": [0, 1, 2, 0, 3, 4],   # 0=Monday, 1=Tuesday, etc.
    "data": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-03", "2024-01-05"]
})

print("\n" + "="*80)
print("EXERCITIU 4: Filtrare Complexa - Query Data")
print("="*80)

print("""
CONTEXT:
Trebuie sa extrageti subset-uri specifice din df_sales.

SARCIINI:
a) Vanzari de Electronics cu valoare_totala > 500
b) Vanzari pe zile lucratoare (zi_saptamana < 5) pentru Laptop si Monitor
c) Top 3 vanzari dupa valoare_totala
d) Media valoare_totala per categorie, doar cele > 800
""")

print("\n--- SOLUTIE ---")


# a) Vanzari de Electronics cu valoare_totala > 500

print("a) Electronics cu valoare_totala > 500:")
mask_a = (df_sales["categorie"] == "Electronics") & (df_sales["valoare_totala"] > 500)
result_a = df_sales.loc[mask_a, ["vanzator", "produs", "valoare_totala"]]
print(result_a)


# b) Vanzari pe zile lucratoare (zi_saptamana < 5) pentru Laptop si Monitor
print("\nb) Vanzari pe zile lucratoare (Mon-Fri) pentru Laptop/Monitor:")
mask_b = (df_sales["zi_saptamana"] < 5) & (df_sales["produs"].isin(["Laptop", "Monitor"]))
result_b = df_sales.loc[mask_b, ["data", "produs", "vanzator"]]
print(result_b)


# c) Top 3 vanzari dupa valoare_totala

print("\nc) Top 3 vanzari dupa valoare_totala:")
top_3 = df_sales.nlargest(3, "valoare_totala")[["vanzator", "produs", "valoare_totala"]]
print(top_3)

# d) Media valoare_totala per categorie, doar cele > 800

print("\nd) Media valoare_totala per categorie (doar > 800):")
medii = df_sales.groupby("categorie")["valoare_totala"].mean()
rezultat_d = medii[medii > 800]
print(rezultat_d)


# a) Vânzări per vânzător
print("a) Vânzări per vânzător:")
sales_per_vendor = df_sales.groupby('vanzator').size().sort_values(ascending=False)
print(sales_per_vendor)

# b) Valoare totală per produs
print("\nb) Valoare totală per produs:")
value_per_product = df_sales.groupby('produs')['valoare_totala'].sum().sort_values(ascending=False)
print(value_per_product)

# c) Min, max, mean preț per produs
print("\nc) Statistici preț per produs:")
price_stats = df_sales.groupby('produs')['pret_unitar'].agg(['min', 'max', 'mean'])
print(price_stats)

# d) Distribuție pe ziua săptămânii
print("\nd) Vânzări per zi săptămână:")
sales_by_day = df_sales.groupby('zi_saptamana').size()
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day_num, count in sales_by_day.items():
    print(f"  {day_names[day_num]}: {count} vânzări")


print("""
CONTEXT:
Aveți df_students cu unii studenți care au nota_test2 lipsă.

SARCINI:
a) Completați nota_test2 lipsă cu media notei pentru acel curs
b) Creați coloană 'nota_medie' = (nota_test1 + nota_test2) / 2
c) Creați coloană 'performanta' bazată pe nota_medie:
   - < 60: 'Slab'
   - 60-75: 'Satisfăcător'
   - > 75: 'Bun'
d) Creați coloană 'risc_pierdere' = 1 dacă absente > 10, altfel 0
e) Raport: Câți studenți per performanta

HINT: fillna() cu groupby, apply(), loc[], astype(int)
""")
# EXERCTITIUL 5
import pandas as pd

# Crearea DataFrame-ului cu datele studenților

df_students = pd.DataFrame({
    "nume": ["Ana", "Mihai", "Ioana", "Alex", "Maria"],
    "curs": ["Python", "Python", "SQL", "Python", "SQL"],
    "nota_test1": [70, 55, 90, 40, 85],
    "nota_test2": [80, None, 95, None, None],
    "absente": [5, 12, 3, 15, 8]
})

print("\n" + "="*80)
print("EXERCITIU 5: Curatare si Feature Engineering")
print("="*80)

print("""
CONTEXT:
Aveti df_students cu unii studenti care au nota_test2 lipsa.

SARCIINI:
a) Completati nota_test2 lipsa cu media notei pentru acel curs
b) Creati coloana 'nota_medie' = (nota_test1 + nota_test2) / 2
c) Creati coloana 'performanta' pe baza notei medii:
   - < 60: 'Slab'
   - 60-75: 'Satisfacator'
   - > 75: 'Bun'
d) Creati coloana 'risc_pierdere' = 1 daca absente > 10, altfel 0
e) Raport: Cati studenti per performanta
""")

print("\n--- SOLUTIE ---")

# a) Completati nota_test2 lipsa cu media notei pentru acel curs

print("a) Completare nota_test2 lipsa cu media pe curs:")

df_students["nota_test2"] = df_students.groupby("curs")["nota_test2"].transform(
    lambda x: x.fillna(x.mean())
)

print(df_students[["nume", "curs", "nota_test2"]])

# b) Creati coloana 'nota_medie' = (nota_test1 + nota_test2) / 2

print("\nb) Creare coloana nota_medie:")

df_students["nota_medie"] = (df_students["nota_test1"] + df_students["nota_test2"]) / 2
print(df_students[["nume", "nota_test1", "nota_test2", "nota_medie"]])


# c) Creati coloana 'performanta' pe baza notei medii:
 #  - < 60: 'Slab'
 #  - 60-75: 'Satisfacator'
 #  - > 75: 'Bun'
print("\nc) Creare coloana performanta:")

def calc_perf(nota):
    if nota < 60:
        return "Slab"
    elif nota <= 75:
        return "Satisfacator"
    else:
        return "Bun"

df_students["performanta"] = df_students["nota_medie"].apply(calc_perf)
print(df_students[["nume", "nota_medie", "performanta"]])


# d) Creati coloana 'risc_pierdere' = 1 daca absente > 10, altfel 0

print("\nd) Creare coloana risc_pierdere:")

df_students["risc_pierdere"] = (df_students["absente"] > 10).astype(int)
print(df_students[["nume", "absente", "risc_pierdere"]])

# e) Raport: Cati studenti per performanta


print("\ne) Raport studenti per performanta:")

raport = df_students["performanta"].value_counts()
print(raport)


# ============================================================================
# EXERCIȚIU 6: Merge + Agregare - Analiza Multi-tabel
# ============================================================================

print("\n" + "="*80)
print("EXERCIȚIU 6: Merge & Multi-table Analysis")
print("="*80)

print("""
CONTEXT:
Aveți două tabele:
- employees: angajați cu depart_id, salariu, ani_exp
- departments: departamente cu buget

SARCINI:
a) Faceți MERGE dintre employees și departments pe depart_id
b) Creați coloană 'salariu_pct_buget' = (salariu / buget_departament) * 100
c) Găsiți departamentul care are cea mai mică medie salariu
d) Top 3 angajați cu cel mai mare raport salariu/buget pe depart
e) Aggregare: Per departament, afișați: nr_angajati, salariu_mediu, salariu_pct_buget_total

HINT: merge(), apply(), groupby(), nlargest()
""")
import pandas as pd

# Crearea DataFrame-urilor pentru employees și departments

employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4, 5],
    "nume": ["Ana", "Mihai", "Ioana", "Alex", "Maria"],
    "depart_id": [10, 20, 10, 30, 20],
    "salariu": [4000, 3500, 5000, 3000, 4500],
    "ani_exp": [2, 5, 7, 1, 4]
})

departments = pd.DataFrame({
    "depart_id": [10, 20, 30],
    "departament": ["IT", "HR", "Finance"],
    "buget": [200000, 150000, 100000]
})




print("\n--- SOLUTIE ---")

# a) Faceți MERGE dintre employees și departments pe depart_id
print("a) Merge employees + departments:")

df = employees.merge(departments, on="depart_id", how="left")
print(df)

# b) Creați coloană 'salariu_pct_buget' = (salariu / buget_departament) * 100
print("\nb) Creare coloana salariu_pct_buget:")

df["salariu_pct_buget"] = (df["salariu"] / df["buget"]) * 100
print(df[["nume", "departament", "salariu", "buget", "salariu_pct_buget"]])

# c) Găsiți departamentul care are cea mai mică medie salariu
print("\nc) Departamentul cu cea mai mica medie salariu:")

medii_salarii = df.groupby("departament")["salariu"].mean()
rezultat_c = medii_salarii.idxmin()
print("Departamentul cu cea mai mica medie salariu este:", rezultat_c)


# d) Top 3 angajați cu cel mai mare raport salariu/buget pe departament
print("\nd) Top 3 angajati cu cel mai mare raport salariu/buget:")

top_3 = df.nlargest(3, "salariu_pct_buget")[["nume", "departament", "salariu_pct_buget"]]
print(top_3)

# e) Aggregare: Per departament, afișați: nr_angajati, salariu_mediu, salariu_pct_buget_total
print("\ne) Aggregare per departament:")

agg = df.groupby("departament").agg(
    nr_angajati=("emp_id", "count"),
    salariu_mediu=("salariu", "mean"),
    salariu_pct_buget_total=("salariu_pct_buget", "sum")
)

print(agg)



        
