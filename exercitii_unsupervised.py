# EXERCITIU 1: Segmentarea Clienților cu K-Means
# ============================================================================

print("=" * 70)
print("EXERCITIU 1: Segmentarea Clienților (Marketing)")
print("=" * 70)
print("""
Sarcină: Generează dataset simulat cu comportamentul clienților.
Antrenează K-Means și folosește elbow method + silhouette.

Dataset: 200 clienți cu 2 feature-uri:
  - Frecvență cumpărări (0-100)
  - Valoare medie tranzacție (0-1000 RON)

TODO:
1. Generează dataset cu make_blobs(200, centers=3, std=15)
2. Antrenează K-Means pentru K=2,3,4,5,6
3. Plotează elbow method
4. Selectează K optim bazat pe elbow + silhouette
5. Afișează segmente și centroizi
""")

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

print("Exercitiu segmentare clienti")

# 1. generare date cu make_blobs
date = make_blobs(
    n_samples=200,                                                         # 200 clienti    
    centers=3,                                                             # 3 segmente diferite
    n_features=2,                                                          # 2 feature-uri: frecventa si valoare
    cluster_std=15,                                                        # dispersie mare pentru a face segmentele mai realiste
    random_state=42                                                        # pentru reproducibilitate  
)

X = date[0]                                                                 # matricea cu cele 200 de randuri si 2 coloane (frecventa, valoare)

# dataframe cu cele 2 coloane cerute
df = pd.DataFrame(X, columns=["Frecventa", "Valoare"])

print("\nPrimele randuri:")
print(df.head())

# 2. testare KMeans pentru K=2..6
k_list = [2, 3, 4, 5, 6]                                                      # lista de valori K pe care le testez
inertii = []                                                                  # lista pentru valorile de inertia (elbow method)
sil = []                                                                      # lista pentru valorile de silhouette (calitatea clusterelor) 

print("\nRezultate KMeans:\n")

for k in k_list:                                                               # pentru fiecare K, antrenez modelul si calculez inertia + silhouette
    km = KMeans(n_clusters=k, random_state=42, n_init=10)                      # antrenez modelul KMeans cu K clusteri, folosind 10 initializari diferite pentru a gasi cea mai buna solutie 
    km.fit(df)                                                                 # antrenez modelul pe datele din dataframe (frecventa si valoare)

    inertii.append(km.inertia_)                                                # adaug valoarea de inertia in lista (pentru elbow method)
    sc = silhouette_score(df, km.labels_)                                      # calculez silhouette score pentru clusterizarea obtinuta (calitatea clusterelor)
    sil.append(sc)                                                             # adaug silhouette score in lista

    print("K =", k, " inertia =", int(km.inertia_), " silhouette =", round(sc, 3))

# 3. alegerea lui K optim (dupa silhouette)
best_k = k_list[sil.index(max(sil))]                                           # aleg K-ul care are cel mai mare silhouette score (cel mai bun cluster)
print("\nK optim este:", best_k)                                               # afisez K-ul optim ales pe baza silhouette score-ului maxim (calitatea clusterelor)

# 4. modelul final
final = KMeans(n_clusters=best_k, random_state=42, n_init=10)                          # antrenez modelul final cu K-ul optim gasit
df["Cluster"] = final.fit_predict(df)                                                  # adaug o coloana noua in dataframe care contine eticheta clusterului pentru fiecare client (0, 1, 2, etc)

# 5. centroizii
centroizi = pd.DataFrame(final.cluster_centers_, columns=["Frecventa", "Valoare"])              #
print("\nCentroizii:")
print(centroizi)

# 6. grafice
plt.figure(figsize=(12,5))                                                              # pentru a afisa doua grafice unul langa altul (elbow method si segmentele clienti)

# elbow
plt.subplot(1,2,1)                                                                              # primul grafic (1 rand, 2 coloane, pozitia 1) pentru elbow method
plt.plot(k_list, inertii, "bo-")                                                                 #  grafic cu puncte albastre conectate prin linii pentru a arata cum scade inertia pe masura ce cresc K
plt.title("Elbow Method")                                                                            # titlu pentru graficul elbow method
plt.xlabel("K")                                                                                   # eticheta pentru axa X (numarul de clusteri)
plt.ylabel("Inertia")                                                                          # eticheta pentru axa Y (inertia - cat de bine se potrivesc punctele in clusteri)

# segmente
plt.subplot(1,2,2)                                                                          # al doilea grafic (1 rand, 2 coloane, pozitia 2) pentru a arata segmentele clienti
plt.scatter(df["Frecventa"], df["Valoare"], c=df["Cluster"], cmap="Set1")                   # grafic de dispersie cu frecventa pe axa X si valoare pe axa Y, colorat in functie de cluster (Set1 este o paleta de culori)
plt.scatter(                                                                                # adaug un punct galben mare pentru a arata pozitia centroizilor in grafic
    centroizi["Frecventa"],
    centroizi["Valoare"],
    c="yellow", s=300, marker="*", edgecolors="black"
)
plt.title("Segmente clienti")                                                                # titlu pentru graficul cu segmentele clienti
plt.xlabel("Frecventa")                                                                      # eticheta pentru axa X (frecventa cumpararilor)
plt.ylabel("Valoare")                                                                        # eticheta pentru axa Y (valoarea medie a tranzactiilor)

plt.tight_layout()                                                                           # pentru a ajusta spatiul dintre grafice astfel incat sa nu se suprapuna
plt.show()                                                                                   # afisez graficele

print("\nGata.")





# ============================================================================
# EXERCITIU 3: Comparație Clustering pe Forme Neliniare
# ============================================================================

print("=" * 70) 
print("EXERCITIU 3: Comparație - K-Means vs DBSCAN pe Forme Neliniare")
print("=" * 70)
print("""
Sarcină: Compare algoritmi pe date cu structuri complexe.

Dataset: make_circles (cercuri concentrice - neliniar!)

TODO:
1. Generează dataset cu make_circles(300, noise=0.05)
2. Antrenează K-Means cu K=2
3. Antrenează DBSCAN cu eps=0.2, min_samples=5
4. Plotează rezultate pe lângă adevar
5. Comentează care algoritm e mai bun și de ce
""")



import pandas as pd
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

print("Exercitiu 3: KMeans vs DBSCAN")

# 1. generez datele cu make_circles cum scrie in cerinta
X, y = make_circles(n_samples=300, noise=0.05, random_state=42)

# pun datele intr-un dataframe ca sa fie mai usor
df = pd.DataFrame(X, columns=["x", "y"])
df["Adevar"] = y

print("\nPrimele randuri:")
print(df.head())

# 2. KMeans cu K=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)                     # antrenez modelul KMeans cu 2 clusteri, folosind 10 initializari diferite pentru a gasi cea mai buna solutie
df["KMeans"] = kmeans.fit_predict(df[["x", "y"]])                             # adaug o coloana noua in dataframe care contine eticheta clusterului pentru fiecare punct (0 sau 1) in functie de rezultatul KMeans

# 3. DBSCAN cu eps=0.2, min_samples=5
db = DBSCAN(eps=0.2, min_samples=5)                                            # antrenez modelul DBSCAN cu o distanta maxima de 0.2 pentru a considera punctele ca fiind in acelasi cluster si cu un minim de 5 puncte pentru a forma un cluster valid
df["DBSCAN"] = db.fit_predict(df[["x", "y"]])                                  # adaug o coloana noua in dataframe care contine eticheta clusterului pentru fiecare punct (0, 1, sau -1 pentru zgomot) in functie de rezultatul DBSCAN  

# 4. grafice
plt.figure(figsize=(15,4))                                                     # pentru a afisa trei grafice unul langa altul (adevar, KMeans, DBSCAN)

# adevar
plt.subplot(1,3,1)                                                             # primul grafic (1 rand, 3 coloane, pozitia 1) pentru a arata adevarul (etichetele reale ale punctelor)
plt.scatter(df["x"], df["y"], c=df["Adevar"], cmap="Set1")                     # grafic de dispersie cu x pe axa X si y pe axa Y, colorat in functie de eticheta reala (Set1 este o paleta de culori)
plt.title("Adevar (make_circles)")                                             # titlu pentru graficul care arata adevarul (structura reala a datelor)

# kmeans
plt.subplot(1,3,2)                                                            # al doilea grafic (1 rand, 3 coloane, pozitia 2) pentru a arata rezultatul KMeans
plt.scatter(df["x"], df["y"], c=df["KMeans"], cmap="Set1")                    # grafic de dispersie cu x pe axa X si y pe axa Y, colorat in functie de eticheta clusterului obtinut de KMeans (Set1 este o paleta de culori)
plt.title("KMeans (K=2)")                                                     # titlu pentru graficul care arata rezultatul KMeans (cum a segmentat punctele in 2 clusteri)

# dbscan
plt.subplot(1,3,3)                                                            # al treilea grafic (1 rand, 3 coloane, pozitia 3) pentru a arata rezultatul DBSCAN
plt.scatter(df["x"], df["y"], c=df["DBSCAN"], cmap="Set1")                    # grafic de dispersie cu x pe axa X si y pe axa Y, colorat in functie de eticheta clusterului obtinut de DBSCAN (Set1 este o paleta de culori)
plt.title("DBSCAN (eps=0.2)")                                                 # titlu pentru graficul care arata rezultatul DBSCAN (cum a segmentat punctele in functie de distanta si densitate)

plt.tight_layout()                                                              # pentru a ajusta spatiul dintre grafice astfel incat sa nu se suprapuna
plt.show()                                                                      # afisez graficele

print("\nComentariu:")
print("KMeans nu merge bine pe forme neliniare (cercuri).")
print("DBSCAN merge mai bine pentru ca detecteaza forme neregulate.")
print("Gata.")
