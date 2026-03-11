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