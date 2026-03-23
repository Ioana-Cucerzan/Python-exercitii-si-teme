import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("MODULE 03 - VIZUALIZARI: EXERCITII PROGRESIVE")
print("=" * 70)

print("\n[EXERCITIU 1] Grafice basice - Familiarizare cu Matplotlib")
print("-" * 70)

print("""
PROBLEMA:
  Incarca dataset Iris. Creeaza 3 grafice separate:
    1. Histogram distributia lungimii sepalei
    2. Scatter plot: lungimea sepalei vs lungimea petalei
    3. Bar plot: numarul de flori pe specie

CERINTE:
  - Figuri separate (nu subplots)
  - Titluri clar, etichete pentru axe
  - Culori diferite pentru fiecare grafic
  - Salveaza: ex1_histogram.png, ex1_scatter.png, ex1_bar.png
""")


# Incarcare dataset
iris = load_iris()                                                           # Incarcam datasetul Iris folosind functia load_iris din sklearn.datasets
df = pd.DataFrame(iris.data, columns=iris.feature_names)                     # Transformam totul intr-un DataFrame Pandas pentru a lucra mai usor.
df["species"] = iris.target                                                  # Adaugam o coloana noua pentru specie, folosind valorile din iris.target.

# 1. Histogram - lungimea sepalei
plt.figure(figsize=(6, 4))                                                   # Setam dimensiunea figurii pentru a fi 6 inch latime si 4 inch inaltime.
plt.hist(df["sepal length (cm)"], bins=20, color="skyblue", edgecolor="black")   #  Creem un histogram pentru coloana "sepal length (cm)", cu 20 de bin-uri, culoare albastru deschis si margini negre.
plt.title("Histogram - Lungimea sepalei")                                         # Adaugam un titlu pentru histogram.
plt.xlabel("Lungime sepal (cm)")                                                  # Adaugam o eticheta pentru axa x.
plt.ylabel("Frecventa")                                                           # Adaugam o eticheta pentru axa y.
plt.savefig("ex1_histogram.png")                                                  # Salvam figura ca ex1_histogram.png in directorul curent.
plt.show()                                                                        # Afisam figura pe ecran. 
#plt.close()                                                                       # Inchidem figura.

# 2. Scatter plot - sepal length vs petal length
plt.figure(figsize=(6, 4))                                                        # Setam dimensiunea figurii pentru a fi 6 inch latime si 4 inch inaltime.
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], color="orange")     # Creem un scatter plot folosind coloana "sepal length (cm)" pentru axa x si "petal length (cm)" pentru axa y, cu puncte de culoare portocalie.
plt.title("Scatter plot - Sepal length vs Petal length")                          # Adaugam un titlu pentru scatter plot.
plt.xlabel("Sepal length (cm)")                                                   # Adaugam o eticheta pentru axa x.
plt.ylabel("Petal length (cm)")                                                   # Adaugam o eticheta pentru axa y.
plt.savefig("ex1_scatter.png")                                                    # Salvam figura ca ex1_scatter.png in directorul curent.
plt.show()                                                                        # Afisam figura pe ecran.
#plt.close()                                                                       # Inchidem figura.

# 3. Bar plot - numar de flori pe specie
species_counts = df["species"].value_counts().sort_index()                        # Calculam numarul de flori pentru fiecare specie folosind value_counts() si sortam indexul pentru a avea ordinea corecta (Setosa, Versicolor, Virginica).

plt.figure(figsize=(6, 4))                                                        # Setam dimensiunea figurii pentru a fi 6 inch latime si 4 inch inaltime.
plt.bar(["Setosa", "Versicolor", "Virginica"], species_counts, color="green")     # Creem un bar plot folosind numele speciilor pe axa x si numarul de flori pe axa y, cu bare de culoare verde.
plt.title("Numar de flori pe specie")                                             # Adaugam un titlu pentru bar plot.
plt.xlabel("Specie")                                                              # Adaugam o eticheta pentru axa x.
plt.ylabel("Numar")                                                               # Adaugam o eticheta pentru axa y. 
plt.savefig("ex1_bar.png")                                                        # Salvam figura ca ex1_bar.png in directorul curent.
plt.show()                                                                        # Afisam figura pe ecran.
#plt.close()                                                                       # Inchidem figura.

print("✓ Exercitiu 1 completat!")
print("   Figuri salvate: ex1_histogram.png, ex1_scatter.png, ex1_bar.png")




import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

print("\n[EXERCITIU 2] Subplots si customizare - Organizare informatii")
print("-" * 70)

print("""
PROBLEMA:
  Creeaza O FIGURA cu 2x2 subplots pentru Iris dataset:
    [0,0] Histogram sepal length
    [0,1] Histogram sepal width
    [1,0] Histogram petal length
    [1,1] Histogram petal width

CERINTE:
  - Un titlu principal pentru figura
  - Fiecare subplot: titlu, etichete axe
  - Culori diferite per subplot
  - tight_layout()
  - Salveaza: ex2_subplots.png
""")

# Incarcare dataset
iris = load_iris()                                                                  # Incarcam datasetul Iris folosind functia load_iris din sklearn.datasets
df = pd.DataFrame(iris.data, columns=iris.feature_names)                            # Transformam totul intr-un DataFrame Pandas pentru a lucra mai usor.            

# Creare figura cu 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))                                     # Creem o figura cu 2 randuri si 2 coloane de subplots, cu dimensiunea totala de 10 inch latime si 8 inch inaltime.

# Subplot [0,0] - sepal length
axes[0, 0].hist(df["sepal length (cm)"], bins=20, color="skyblue", edgecolor="black")               # Creem un histogram pentru coloana "sepal length (cm)" in subplotul [0,0], cu 20 de bin-uri, culoare albastru deschis si margini negre.
axes[0, 0].set_title("Sepal length")                                                                 # Adaugam un titlu pentru subplotul [0,0].
axes[0, 0].set_xlabel("cm")                                                                         # Adaugam o eticheta pentru axa x in subplotul [0,0].
axes[0, 0].set_ylabel("Frecventa")                                                                      # Adaugam o eticheta pentru axa y in subplotul [0,0].

# Subplot [0,1] - sepal width
axes[0, 1].hist(df["sepal width (cm)"], bins=20, color="orange", edgecolor="black")                     # Creem un histogram pentru coloana "sepal width (cm)" in subplotul [0,1], cu 20 de bin-uri, culoare portocalie si margini negre.
axes[0, 1].set_title("Sepal width")                                                                # Adaugam un titlu pentru subplotul [0,1].
axes[0, 1].set_xlabel("cm")                                                                         # Adaugam o eticheta pentru axa x in subplotul [0,1].
axes[0, 1].set_ylabel("Frecventa")                                                                      # Adaugam o eticheta pentru axa y in subplotul [0,1].

# Subplot [1,0] - petal length
axes[1, 0].hist(df["petal length (cm)"], bins=20, color="green", edgecolor="black")                    # Creem un histogram pentru coloana "petal length (cm)" in subplotul [1,0], cu 20 de bin-uri, culoare verde si margini negre.
axes[1, 0].set_title("Petal length")                                                                        # Adaugam un titlu pentru subplotul [1,0].
axes[1, 0].set_xlabel("cm")                                                                              # Adaugam o eticheta pentru axa x in subplotul [1,0].
axes[1, 0].set_ylabel("Frecventa")                                                                       # Adaugam o eticheta pentru axa y in subplotul [1,0].

# Subplot [1,1] - petal width
axes[1, 1].hist(df["petal width (cm)"], bins=20, color="purple", edgecolor="black")                         # Creem un histogram pentru coloana "petal width (cm)" in subplotul [1,1], cu 20 de bin-uri, culoare mov si margini negre.
axes[1, 1].set_title("Petal width")                                                                       # Adaugam un titlu pentru subplotul [1,1].
axes[1, 1].set_xlabel("cm")                                                                            # Adaugam o eticheta pentru axa x in subplotul [1,1].
axes[1, 1].set_ylabel("Frecventa")                                                                          # Adaugam o eticheta pentru axa y in subplotul [1,1].

# Titlu principal
fig.suptitle("Iris Dataset - Distributii ale caracteristicilor", fontsize=16)

# Aranjare automata
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Salvare
plt.savefig("ex2_subplots.png")

# Afisare
plt.show()

print("✓ Exercitiu 2 completat!")
print("   Figura salvata: ex2_subplots.png")




import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

print("\n[EXERCITIU 3] Analiza corelatie - EDA esential")
print("-" * 70)

print("""
PROBLEMA:
  Creeaza DOUA figuri:
    1. Heatmap matrice corelatie Iris (seaborn)
    2. Pairplot Iris cu hue='species'

CERINTE:
  - Heatmap: annotare valorile, cmap='coolwarm', center=0
  - Pairplot: diag_kind='hist', hue='species'
  - ANALITICA: In comentarii, ce patterns observi? Ce features sunt corelate?
  - Salveaza: ex3_heatmap.png, ex3_pairplot.png
""")

# Incarcare dataset
iris = load_iris()                                                             # Incarcam datasetul Iris folosind functia load_iris din sklearn.datasets
df = pd.DataFrame(iris.data, columns=iris.feature_names)                      # Transformam totul intr-un DataFrame Pandas pentru a lucra mai usor.
df["species"] = iris.target                                                   # Adaugam o coloana noua pentru specie, folosind valorile din iris.target.

# 1. HEATMAP CORELATIE
plt.figure(figsize=(8, 6))                                                    # Setam dimensiunea figurii pentru a fi 8 inch latime si 6 inch inaltime.
corr = df.drop(columns=["species"]).corr()                                       # Calculam matricea de corelatie pentru variabilele numerice, eliminand coloana "species".

sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)                       # Creem un heatmap folosind seaborn, cu valorile corelatiei annotate in celule, folosind colormap-ul "coolwarm" si centrul la 0 pentru a evidentia corelatiile pozitive si negative.
plt.title("Matrice de corelatie - Iris")                                      # Adaugam un titlu pentru heatmap.
plt.savefig("ex3_heatmap.png")                                                  # Salvam figura ca ex3_heatmap.png in directorul curent.
plt.show()

# 2. PAIRPLOT
sns.pairplot(df, hue="species", diag_kind="hist")                              # Creem un pairplot folosind seaborn, cu variabilele din DataFrame, colorand punctele in functie de specie (hue="species") si folosind histograme pentru diagona (diag_kind="hist").
plt.savefig("ex3_pairplot.png")                                                # Salvam figura ca ex3_pairplot.png in directorul curent.
plt.show()

"""
ANALIZA EDA (interpretare):

1. Corelatii puternice:
   - petal length si petal width au corelatie foarte mare (aproape 0.96)
   - sepal length are corelatie moderata cu petal length si petal width
   - sepal width este cel mai slab corelat cu restul variabilelor

2. Observatii din pairplot:
   - Setosa este complet separata de celelalte doua specii pe petal length si petal width
   - Versicolor si Virginica se suprapun partial, dar tot exista separare pe petal length
   - Sepal width nu separa bine speciile
   - Petal length este cel mai bun feature pentru clasificare

Concluzie:
  Petal length si petal width sunt cele mai informative variabile pentru clasificarea speciilor Iris.
"""
print("✓ Exercitiu 3 completat!")
print("   Figuri salvate: ex3_heatmap.png, ex3_pairplot.png")



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

print("\n[EXERCITIU 4] Evaluare model - Diagnosticare ML")
print("-" * 70)

print("""
PROBLEMA:
  Antreneaza RandomForest pe Iris dataset (3-class classification).
  Creeaza DOUA figuri:
    1. Confusion matrix (heatmap) pentru test set
    2. Feature importance (bar plot horizontal)

CERINTE:
  - Confusion matrix: annot=True, classes=['setosa', 'versicolor', 'virginica']
  - Feature importance: ordonat descrescator, barh (orizontal)
  - Salveaza: ex4_confusion_matrix.png, ex4_feature_importance.png

BONUS: In comentarii, ce iti spune confusion matrix despre model?
""")

# Incarcare dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Impartire train/test
X_train, X_test, y_train, y_test = train_test_split(                                                 # Impartim datele in set de antrenare si testare folosind functia train_test_split din sklearn.model_selection, cu 30% pentru testare si random_state=42 pentru reproducibilitate.
    X, y, test_size=0.3, random_state=42
)

# Antrenare model
model = RandomForestClassifier(random_state=42)                                                        # Cream un model de clasificare RandomForest folosind RandomForestClassifier din sklearn.ensemble, cu random_state=42 pentru reproducibilitate.
model.fit(X_train, y_train)                                                                            # Antrenam modelul folosind metoda fit() pe datele de antrenare (X_train, y_train).

# Predictii
y_pred = model.predict(X_test)                                                                          # Facem predictii folosind metoda predict() pe setul de testare (X_test) si stocam rezultatele in y_pred.

# 1. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)                                                               # Calculam matricea de confuzie folosind functia confusion_matrix din sklearn.metrics, comparand valorile reale (y_test) cu cele prezise (y_pred).

plt.figure(figsize=(6, 5))                                                                              # Setam dimensiunea figurii pentru a fi 6 inch latime si 5 inch inaltime
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix - RandomForest")                                                           # Adaugam un titlu pentru heatmap-ul matricei de confuzie.
plt.xlabel("Predicted")                                                                                # Adaugam o eticheta pentru axa x a heatmap-ului matricei de confuzie.
plt.ylabel("True")                                                                                      # Adaugam o eticheta pentru axa y a heatmap-ului matricei de confuzie.
plt.savefig("ex4_confusion_matrix.png")                                                                # Salvam figura ca ex4_confusion_matrix.png in directorul curent.
plt.show()

# 2. FEATURE IMPORTANCE
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # sortare descrescatoare

plt.figure(figsize=(8, 5))                                                                        # Setam dimensiunea figurii pentru a fi 8 inch latime si 5 inch inaltime.
plt.barh(np.array(feature_names)[indices], importances[indices], color="green")                     # Creem un bar plot orizontal folosind plt.barh, cu numele caracteristicilor sortate descrescator pe axa y si valorile importantei pe axa x, cu bare de culoare verde.
plt.title("Feature Importance - RandomForest")                                                    # Adaugam un titlu pentru bar plot-ul importantei caracteristicilor.
plt.xlabel("Importance")                                                                            # Adaugam o eticheta pentru axa x a bar plot-ului importantei caracteristicilor.
plt.ylabel("Feature")                                                                            # Adaugam o eticheta pentru axa y a bar plot-ului importantei caracteristicilor.
plt.gca().invert_yaxis()                                                                      # cel mai important sus
plt.savefig("ex4_feature_importance.png")
plt.show()


# EXERCITIU 5: CHALLENGE - RAPORT COMPLET
# ==============================================================================

print("\n[EXERCITIU 5] CHALLENGE - Raport vizualizare complet")
print("-" * 70)

print("""
PROBLEMA:
  Lucreaza cu dataset CUSTOM (binary classification, 300 samples, 5 features).
  Creeaza UN RAPORT cu 6 grafice intr-o FIGURA 2x3:
    [0,0] Class distribution (countplot)
    [0,1] Feature 1 vs Feature 2 scatter
    [0,2] Correlation heatmap
    [1,0] Confusion matrix
    [1,1] ROC curve
    [1,2] Feature importance

CERINTE:
  - Antrenaza RandomForest
  - Layout profesional (suptitle, tight_layout)
  - Fiecare subplot: titlu, axe etichetate
  - Culori consistente
  - Salveaza: ex5_raport_complet.png

BONUS: Putati folosi subplots impreuna? (hint: nu direct, csv separate)
""")



# Rezolvare Laurentiu - 2 figuri separate (matplotlib subplots + seaborn au conflicte)

# TODO: SCRIE CODUL AICI
# ============ INCEPUT ============

# Genereaza dataset
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc

X_custom, y_custom = make_classification(
    n_samples=300, n_features=5, n_informative=4, n_redundant=1,
    random_state=42
)
df_custom = pd.DataFrame(X_custom, columns=[f'Feature_{i}' for i in range(5)])
df_custom['target'] = y_custom

# Antrenare model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_custom, y_custom, test_size=0.2, random_state=42
)
rf_custom = RandomForestClassifier(n_estimators=100, random_state=42)
rf_custom.fit(X_train_c, y_train_c)
y_pred_c = rf_custom.predict(X_test_c)
y_pred_proba_c = rf_custom.predict_proba(X_test_c)[:, 1]

# NOTA: Din limitari tehnice, facem 2 figuri separate
# (matplotlib subplots + seaborn au conflicte)

# Figura 1: Grafice statice (3 subplots)
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4))

# [0,0] Class distribution
class_counts = pd.Series(y_custom).value_counts()
axes1[0].bar(['Class 0', 'Class 1'], class_counts.values, color=['skyblue', 'salmon'], alpha=0.7, edgecolor='black')
axes1[0].set_ylabel('Numar', fontsize=10, fontweight='bold')
axes1[0].set_title('Class Distribution', fontsize=11, fontweight='bold')
for i, v in enumerate(class_counts.values):
    axes1[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# [0,1] Feature scatter
axes1[1].scatter(df_custom['Feature_0'][y_custom==0], df_custom['Feature_1'][y_custom==0],
                 label='Class 0', alpha=0.6, s=50, color='skyblue', edgecolors='black', linewidth=0.5)
axes1[1].scatter(df_custom['Feature_0'][y_custom==1], df_custom['Feature_1'][y_custom==1],
                 label='Class 1', alpha=0.6, s=50, color='salmon', edgecolors='black', linewidth=0.5)
axes1[1].set_xlabel('Feature 0', fontsize=10, fontweight='bold')
axes1[1].set_ylabel('Feature 1', fontsize=10, fontweight='bold')
axes1[1].set_title('Feature 0 vs Feature 1', fontsize=11, fontweight='bold')
axes1[1].legend(fontsize=9)
axes1[1].grid(True, alpha=0.3)

# [0,2] Correlation heatmap
corr = df_custom.iloc[:, :-1].corr()
im = axes1[2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes1[2].set_xticks(range(5))
axes1[2].set_yticks(range(5))
axes1[2].set_xticklabels([f'F{i}' for i in range(5)], fontsize=9)
axes1[2].set_yticklabels([f'F{i}' for i in range(5)], fontsize=9)
axes1[2].set_title('Correlation Matrix', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=axes1[2], fraction=0.046, pad=0.04)

plt.suptitle('Raport Complet (Parte 1: Explorare)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('ex5_raport_p1.png', dpi=150, bbox_inches='tight')
plt.close()

# Figura 2: Grafice model (3 subplots)
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))

# [1,0] Confusion matrix
cm_custom = confusion_matrix(y_test_c, y_pred_c)
im1 = axes2[0].imshow(cm_custom, cmap='Blues')
axes2[0].set_xticks([0, 1])
axes2[0].set_yticks([0, 1])
axes2[0].set_xticklabels(['Class 0', 'Class 1'], fontsize=10)
axes2[0].set_yticklabels(['Class 0', 'Class 1'], fontsize=10)
axes2[0].set_ylabel('True', fontsize=10, fontweight='bold')
axes2[0].set_xlabel('Predicted', fontsize=10, fontweight='bold')
axes2[0].set_title('Confusion Matrix', fontsize=11, fontweight='bold')
for i in range(2):
    for j in range(2):
        axes2[0].text(j, i, str(cm_custom[i, j]), ha='center', va='center',
                     color='white' if cm_custom[i, j] > cm_custom.max()/2 else 'black',
                     fontweight='bold', fontsize=12)
plt.colorbar(im1, ax=axes2[0], fraction=0.046, pad=0.04)

# [1,1] ROC curve
fpr_c, tpr_c, _ = roc_curve(y_test_c, y_pred_proba_c)
roc_auc_c = auc(fpr_c, tpr_c)
axes2[1].plot(fpr_c, tpr_c, 'o-', color='darkorange', linewidth=2, label=f'AUC={roc_auc_c:.3f}')
axes2[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes2[1].fill_between(fpr_c, tpr_c, alpha=0.2, color='darkorange')
axes2[1].set_xlim([0, 1])
axes2[1].set_ylim([0, 1.05])
axes2[1].set_xlabel('FPR', fontsize=10, fontweight='bold')
axes2[1].set_ylabel('TPR', fontsize=10, fontweight='bold')
axes2[1].set_title('ROC Curve', fontsize=11, fontweight='bold')
axes2[1].legend(fontsize=9)
axes2[1].grid(True, alpha=0.3)

# [1,2] Feature importance
importance = rf_custom.feature_importances_
feat_names = [f'F{i}' for i in range(5)]
sorted_idx = np.argsort(importance)[::-1]
axes2[2].barh(range(len(sorted_idx)), importance[sorted_idx],
              color=plt.cm.viridis(np.linspace(0.3, 0.9, 5)))
axes2[2].set_yticks(range(len(sorted_idx)))
axes2[2].set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=10)
axes2[2].set_xlabel('Importance', fontsize=10, fontweight='bold')
axes2[2].set_title('Feature Importance', fontsize=11, fontweight='bold')
axes2[2].grid(True, alpha=0.3, axis='x')

plt.suptitle('Raport Complet (Partea 2: Model Evaluation)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('ex5_raport_p2.png', dpi=150, bbox_inches='tight')
plt.close()