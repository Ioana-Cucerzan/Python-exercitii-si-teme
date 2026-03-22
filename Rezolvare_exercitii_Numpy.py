#Calculeaza norma L2 (euclidiana) a unui vector

import numpy as np 
vector = np.array([3, 4, 5])  # exemplu de vector
patrate = vector ** 2    # ridicarea la patrat a fiecarui element
suma = np.sum(patrate)   # suma elementelor patrate
radacina_patrata = np.sqrt(suma)  # calcularea radacinii patrate a sumei
print("Norma L2 (euclidiana) a vectorului este:", radacina_patrata)
print("Suma patratelor este:" , suma)
print("Patratele elementelor vectorului sunt:" , patrate)



# EXERCITIU:
"""
Creeaza un array cu numerele 1-10 (inclusiv).
Apoi raspunde la intrebarile:
1. Care e shape-ul?
2. Cati elemente are?
3. Care e al 5-lea element (indexul 4)?
4. Care sunt ultimele 3 elemente?"""

# Crearea array-ului cu numerele de la 1 la 10
import numpy as np
import time

print("=" * 80)
print("EXERCITII NUMPY - MODULE 01")
print("=" * 80)
print("\n" + "=" * 80)
print("EXERCITIU 1: Crearea si Explorarea Array-urilor (USOR)")
print("=" * 80)


# Crearea array-ului cu numerele de la 1 la 10
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#1. Care este shape-ul?
print("1. Shape-ul array-ului este:", arr.shape)
#2. Cate elemente are?
print("2. Numarul de elemente din array este:", arr.size)
#3. Care este al 5-lea element (indexul 4)?
print("3. Al 5-lea element (indexul 4) este:", arr[4])
#4. Care sunt ultimele 3 elemente?
print("4. Ultimele 3 elemente sunt:", arr[-3:])



"""
Se da o matrice 3x4:
[[1,  2,  3,  4],
 [5,  6,  7,  8],
 [9, 10, 11, 12]]

Extrage:
1. Elementul din randul 1, coloana 2 (ar trebui 7)
2. Intreaga coloana 3 (ar trebui [3, 7, 11])
3. Ultimul rand (ar trebui [9, 10, 11, 12])
4. Primele doua randuri, primele doua coloane
"""

print("\nEXERCITIU 2:")
print("Extrage elemente din matrice!")

# Crearea matricei 3x4
matrice = np.array([[1,  2,  3,  4],
                    [5,  6,  7,  8],    
                    [9, 10, 11, 12]])       
#1. Elementul din randul 1, coloana 2 (ar trebui 7)
print("1. Elementul din randul 1, coloana 2 este:", matrice[1, 2])

#2. Intreaga coloana 3 (ar trebui [3, 7, 11])
print("2. Intreaga coloana 3 este:", matrice[:, 2])

#3. Ultimul rand (ar trebui [9, 10, 11, 12])
print("3. Ultimul rand este:", matrice[-1, :])

#4. Primele doua randuri, primele doua coloane
print("4. Primele doua randuri, primele doua coloane sunt:", matrice[:2, :2])


"""
Se da un array cu 12 elemente: [0, 1, 2, ..., 11]

1. Reshapeaza-l in matrice 3x4
2. Reshapeaza-l in matrice 4x3
3. Reshapeaza-l inapoi in 1D array
4. Afla ce se intampla cu arr.reshape(-1)
"""

print("\nEXERCITIU 3:")
print("Joaca cu reshape si shape!")

# Crearea array-ului cu 12 elemente
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

#1. Reshapeaza-l in matrice 3x4
mattrix_3x4 = arr.reshape(3, 4)
print("1. Matricea reshaped in 3x4 este:\n", mattrix_3x4)

#2. Reshapeaza-l in matrice 4x3
mattrix_4x3 = arr.reshape(4, 3) 
print("2. Matricea reshaped in 4x3 este:\n", mattrix_4x3)

#3. Reshapeaza-l inapoi in 1D array
arr_1d = arr.reshape(-1)
print("3. Array-ul reshaped inapoi in 1D este:\n", arr_1d)

#4. Afla ce se intampla cu arr.reshape(-1)  -> calculeaza automat numarul de elemente si reshapeaza in 1D array
print("4. Ce se intampla cu arr.reshape(-1)?")
print("   ", arr.reshape(-1))


"""
Se dau doi vectori:
a = [1, 2, 3]
b = [4, 5, 6]

1. Calculeaza a * b (inmultire element-wise)
2. Calculeaza dot product: a • b
3. Care e diferenta?
4. CHALLENGE: Calculeaza dot product manual (fara np.dot)"""

print("\nEXERCITIU 4:")
print("Intelege diferenta intre inmultire element-wise si dot product!")

# Crearea vectorilor
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

#1. Calculeaza a * b (inmultire element-wise)
element_wise = a * b
print("1. Inmultirea element-wise a * b este:", element_wise)

#2. Calculeaza dot product: a • b
dot_product = np.dot(a,b)
print("2. Dot product a • b este:", dot_product)

#3. Care e diferenta?
print("3. Diferenta este ca inmultirea element-wise inmulteste fiecare element corespunzator, in timp ce dot product aduna rezultatele inmultirii elementelor corespunzatoare.")


# EXERCITIU:
"""
Se dau urmatoarele scoruri: [45, 78, 92, 34, 88, 56, 91, 45, 89]

1. Extrage toate scorurile mai mari decat 75 (promovati)
2. Extrage toate scorurile intre 50 si 80 (incercati mai mult)
3. Numara cati au picat (<50)
4. Calculeaza media scorurilor promovate
"""
scores = [45, 78, 92, 34, 88, 56, 91, 45, 89]

# 1. Scoruri > 75 (promovati)
promovati = [s for s in scores if s > 75]
print("1) Promovati:", promovati)

# 2. Scoruri intre 50 si 80
interval_50_80 = [s for s in scores if 50 <= s <= 80]
print("2) Scoruri intre 50 si 80:", interval_50_80)

# 3. Cati au picat (<50)
picati = len([s for s in scores if s < 50])
print("3) Numar picati:", picati)

# 4. Media scorurilor promovate
media_promovati = sum(promovati) / len(promovati)
print("4) Media promovati:", media_promovati)



# EXERCITIU:6
"""
Se da o matrice de temperatura: 3 locatii x 4 zile
[[20, 22, 21, 23],   (Bucuresti)
 [15, 17, 16, 18],   (Cluj)
 [18, 19, 20, 21]]   (Constanta)

Scade media temperaturii fiecarei locatii din toate valorile.

Hint: Calculeaza media pe axis=1 (pentru fiecare locatie)
      Apoi scade cu broadcasting
"""
import numpy as np

# Matricea temperaturilor
temps = np.array([
    [20, 22, 21, 23],   # Bucuresti
    [15, 17, 16, 18],   # Cluj
    [18, 19, 20, 21]    # Constanta
])

# 1. Media temperaturilor pe fiecare locatie (axis=1)
medii = temps.mean(axis=1)

# 2. Scaderea mediei din fiecare rand (broadcasting)
rezultat = temps - medii[:, None]

print("Matrice originala:\n", temps)
print("\nMedii pe locatie:", medii)
print("\nRezultat dupa scaderea mediilor:\n", rezultat)


# ============================================================================
# EXERCITIU 7: Vectorizare vs For Loop (GREU - Performanta)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCITIU 7: Vectorizare vs For Loop - Viteza (GREU)")
print("=" * 80)

# EXERCITIU:
"""
Se dau doi vectori cu 1 milion de elemente.

1. Calculeaza distanta euclidiana cu for loop
2. Calculeaza distanta euclidiana cu NumPy vectorizat
3. Compara timpii de executie
4. Cat de mult mai rapid e NumPy?

Hint: sqrt(sum((a-b)^2))
"""

print("\nEXERCITIU 7:")
print("Compara viteza: for loop vs vectorizare NumPy!")

# INTUITIE: Aceasta e MOTIVATIA principala pentru NumPy!

import time
# Generare vectori random
size = 10**6
a = np.random.rand(size)
b = np.random.rand(size)    
# 1. Distanta euclidiana cu for loop
start_time = time.time()
euclidean_distance_loop = np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(size)))
end_time = time.time()  
print(f"Distanta euclidiana (for loop): {euclidean_distance_loop:.4f}, Timp: {end_time - start_time:.4f} sec")
# 2. Distanta euclidiana cu NumPy vectorizat    
start_time = time.time()
euclidean_distance_numpy = np.sqrt(np.sum((a - b) ** 2))
end_time = time.time()
print(f"Distanta euclidiana (NumPy): {euclidean_distance_numpy:.4f}, Timp: {end_time - start_time:.4f} sec")
# 3. Comparatie timpi de executie
time_loop = end_time - start_time
time_numpy = end_time - start_time  
print(f"Viteza NumPy vs For Loop: {time_loop / time_numpy:.2f}x mai rapid!")    



# ============================================================================
# EXERCITIU 8: Distanta Euclidiana intre Puncte (GREU - Aplicat)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCITIU 8: Distanta Euclidiana - KNN Preparation (GREU)")
print("=" * 80)

# EXERCITIU:
"""
Se dau urmatoarele puncte in 2D:
A = (1, 2)
B = (4, 6)
C = (7, 3)
D = (2, 8)

1. Calculeaza distantele dintre A si toate celelalte puncte
2. Gaseste punctul cel mai apropiat de A
3. CHALLENGE: Creeaza o functie care calculeaza distanta
               intre doua puncte in n dimensiuni
"""

print("\nEXERCITIU 8:")
print("Aplicare: Calcula distante (util pentru KNN, clustering)!")

# INTUITIE: Aceasta e o operatie fundamentala in ML!

# Definirea punctelor
A = np.array([1, 2])    
B = np.array([4, 6])
C = np.array([7, 3])
D = np.array([2, 8])
points = [B, C, D]
# 1. Calculeaza distantele dintre A si celelalte puncte
distances = [np.sqrt(np.sum((A - p) ** 2)) for p in points]
print("Distante de la A la B, C, D:", distances)
# 2. Gaseste punctul cel mai apropiat de A
closest_point_index = np.argmin(distances)
closest_point = points[closest_point_index]
print(f"Punctul cel mai apropiat de A este: {closest_point} cu distanta {distances[closest_point_index]:.4f}")
# 3. CHALLENGE: Functie pentru distanta in n dimensiuni
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
# Testare functie
print("\nTestare functie euclidean_distance:")
print("Distanta A-B:", euclidean_distance(A, B))
print("Distanta A-C:", euclidean_distance(A, C))
print("Distanta A-D:", euclidean_distance(A, D))



# ============================================================================
# EXERCITIU 9: Normalizare (Standardizare) (GREU - Aplicat)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCITIU 9: Normalizare - Feature Scaling (GREU)")
print("=" * 80)

# EXERCITIU:
"""
Se da un dataset cu doua feature-uri (coloane):
Feature 1 (Varsta): [20, 25, 30, 35, 40]
Feature 2 (Salariu): [25000, 30000, 35000, 40000, 45000]

1. Normalizeaza fiecare feature (mean=0, std=1)
2. Verifica ca media e ~0 si std e ~1
3. Observa cum tabelul arata dupa normalizare
4. CHALLENGE: Implementeaza o functie de normalizare

NOTA: Aceasta e ESENTIALA in ML! Algoritmi ca KNN, SVM, Neural Networks
      au nevoie de date normalizate pentru a converge mai rapid
"""

print("\nEXERCITIU 9:")
print("Normalizare (Feature Scaling) - ESENTIALA in ML!")

# INTUITIE: Normalizarea pune toate feature-urile pe aceasi scara
# Definirea dataset-ului
ages = np.array([20, 25, 30, 35, 40])   
salaries = np.array([25000, 30000, 35000, 40000, 45000])
# 1. Normalizarea fiecarui feature
ages_normalized = (ages - ages.mean()) / ages.std()
salaries_normalized = (salaries - salaries.mean()) / salaries.std()
# 2. Verificare media si std    
print("Media varste normalizate:", ages_normalized.mean())
print("Std varste normalizate:", ages_normalized.std())
print("Media salarii normalizate:", salaries_normalized.mean())
print("Std salarii normalizate:", salaries_normalized.std())
# 3. Tabelul dupa normalizare
print("\nTabel normalizat:")
print("Varsta Normalizata:", ages_normalized)
print("Salariu Normalizat:", salaries_normalized)   
# 4. CHALLENGE: Functie de normalizare
def normalize(feature):
    return (feature - feature.mean()) / feature.std()
# Testare functie
print("\nTestare functie normalize:")   


# ============================================================================
# EXERCITIU 10: Agregari - Statistici (GREU - Aplicat)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCITIU 10: Agregate si Statistici (GREU)")
print("=" * 80)

# EXERCITIU:
"""
Se dau rezultatele testelor pentru 4 studenti x 5 teste:

Student 1: [85, 90, 92, 88, 91]
Student 2: [78, 82, 80, 81, 79]
Student 3: [95, 93, 96, 94, 95]
Student 4: [70, 72, 71, 73, 70]

1. Calculeaza media pentru fiecare student
2. Calculeaza media pentru fiecare test
3. Care student are cea mai buna medie?
4. Care test a fost cel mai greu (media cea mai mica)?
5. Care e media generala a clasei?
"""

print("\nEXERCITIU 10:")
print("Agregate si statistici pe date reale!")

# INTUITIE: Aceasta e cum analizeaza datele in ML - cu agregate!
# Definirea dataset-ului
students_scores = np.array([    
    [85, 90, 92, 88, 91],   # Student 1
    [78, 82, 80, 81, 79],   # Student 2
    [95, 93, 96, 94, 95],   # Student 3
    [70, 72, 71, 73, 70]    # Student 4
])
# 1. Media pentru fiecare student
students_means = students_scores.mean(axis=1)
print("Media pentru fiecare student:", students_means)
# 2. Media pentru fiecare test
tests_means = students_scores.mean(axis=0)
print("Media pentru fiecare test:", tests_means)    
# 3. Studentul cu cea mai buna medie
best_student_index = np.argmax(students_means)
print(f"Studentul cu cea mai buna medie este Student {best_student_index + 1} cu media {students_means[best_student_index]:.2f}")
# 4. Testul cel mai greu (media cea mai mica)   
hardest_test_index = np.argmin(tests_means)
print(f"Testul cel mai greu este Testul {hardest_test_index + 1} cu media {tests_means[hardest_test_index]:.2f}")
# 5. Media generala a clasei
class_mean = students_scores.mean()
print(f"Media generala a clasei este: {class_mean:.2f}")    
