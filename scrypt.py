# Este archivo se divide en tres secciones principales: 
# análisis exploratorio (desde la línea 33), clasificación binaria (desde la línea 145) 
# y clasificación multiclase (desde la línea 470). En el mismo se implementan visualizaciones 
# de los píxeles más informativos, entrenamiento y evaluación de modelos supervisados como 
# k-Nearest Neighbors, Árboles de Decisión y Regresión Logística, con selección de atributos 
# mediante varianza, información mutua y métodos aleatorios. Además, se utilizan técnicas de 
# validación como train/test split, hold-out y validación cruzada con búsqueda de hiperparámetros 
# por GridSearch. El archivo incluye tanto métricas clásicas (accuracy, precisión, recall, F1-score, 
# AUC) como gráficas de matriz de confusión, heatmaps y comparación visual de modelos.


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score,precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns


#%%
#Cargar el dataset
df = pd.read_csv('poner path correspondiente!!!') #Cambiar path al correspondiente


##############################
##  ANALISIS EXPLORATORIO   ##
##############################

#%%
pixel_cols = [c for c in df.columns if c != "label" and not c.startswith("Unnamed")] #sacamos la columna de label
X = df[pixel_cols].astype("float32") / 255.0        #normalizamos los valores de los pixeles
y = df["label"].to_numpy()                          #columna de labels

#%%
#Grafico 1.a_varianza
var_scores = X.var(axis=0).to_numpy()    #varianza respecto al promedio de cada pixel            

k = 100                                              
top_idx = np.argsort(var_scores)[-k:][::-1]         #muestra los 100 pixeles que mas varian

mask_top = np.zeros(784, dtype=bool)          #array booleana de False
mask_top[top_idx] = True                      #convierte en True a los top 100
mask_top = mask_top.reshape(28, 28)           #arma una matriz de 28x28 a partir del array anterior


mean_global = X.mean(axis=0).to_numpy().reshape(28, 28)      #promedio por columna de cada pixel 

#graficamos superponiendo las dos matrices
fig1= plt.figure(figsize=(4, 4))
plt.imshow(mean_global, cmap="gray")                # fondo(promedio)
plt.imshow(mask_top, cmap="Blues", alpha=0.75)      # píxeles top-k en azul
plt.title(f"Píxeles más informativos (top {k})")
plt.axis("off")
plt.tight_layout()
plt.show()

#%%
#Grafico 1a_mutual_information
# Calcular la información mutua entre cada pixel (columna de X) y la clase y
mi_scores = mutual_info_classif(X, y, discrete_features=True)  # y debe ser un vector con las clases

k = 100
top_idx = np.argsort(mi_scores)[-k:][::-1]  # índices de los k píxeles con mayor info mutua

mask_top = np.zeros(784, dtype=bool)
mask_top[top_idx] = True
mask_top = mask_top.reshape(28, 28)

mean_global = X.mean(axis=0).to_numpy().reshape(28, 28)  # imagen promedio

# Graficar
fig1 = plt.figure(figsize=(4, 4))
plt.imshow(mean_global, cmap="gray")                     # fondo: imagen promedio
plt.imshow(mask_top, cmap="Blues", alpha=0.75)           # píxeles informativos en azul
plt.title(f"Píxeles con mayor mutual info (top {k})")
plt.axis("off")
plt.tight_layout()
plt.show()

#%%
#Grafico 1b_correlacion_promedios
means = np.zeros((10, 784), dtype=np.float32)  #matriz de ceros

for cls in range(10):
    means[cls] = X[y == cls].mean(axis=0)   #calcula el promedio de cada pixel por clase


corr = np.corrcoef(means)        #matriz de 10x10 con la relacion de variacion entre los promedios de cada clase, los mas cercanos a uno son los que mas se parecen

#hacemos un heatmap de esa matriz
fig2= plt.figure(figsize=(4.5, 4))
sns.heatmap(corr, annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1,
            xticklabels=range(10), yticklabels=range(10))
plt.title("Correlación entre promedios por clase")
plt.tight_layout()
plt.show()

#%%
#Grafico 1.c heatmaps_clases
#armamos una lista con las clases que queremos mostrar
clases = [0,5]

#para cada clase hacemos un grafico
for clase in clases:
    X_plot = X[y == clase]                    #Seleccionamos las filas que pertenecen a la clase

    var_img = X_plot.var(axis=0).values.reshape(28, 28)     #varianza respecto al promedio de cada pixel y reshape para matriz de 28x28


    #Graficamos un heatmap
    fig3, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(var_img)

    #Etiquetado de pixeles en ambos ejes
    pixeles = np.arange(28)
    ax.set_xticks(pixeles)    
    ax.set_yticks(pixeles)
    ax.set_xticklabels(pixeles,fontsize=7)
    ax.set_yticklabels(pixeles,fontsize=7)

    #Añadimos cuadrícula para distinguir los pixeles entre si
    ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.2)


    ax.set_xlabel("Columna de píxel")
    ax.set_ylabel("Fila de píxel")
    ax.set_title(f"Heatmap de varianza de la clase {clase}")
    plt.colorbar(img, ax=ax, label="Varianza")

    plt.tight_layout()
    plt.show()


##############################
##  CLASIFICACION BINARIA   ##
##############################

#%%
# Filtrar las clases 0 y 8
df_0_8 = df[df["label"].isin([0, 8])]

print("cantaidad de cada clase", df_0_8["label"].value_counts())
#%%

# Separar datos en features y labels
X = df_0_8.drop("label", axis=1)
Y = df_0_8["label"]

random_state=42  #semilla de random

# Separar en train 80% y test 20%. stratify me balancea la cantidad de clases en los set, y random_state es la semilla
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=random_state
)
#%%

# Configuraciones iniciales
ks = [5, 10, 15, 20, 25, 30, 60] # la cantidad de k pixeles para analizar
pixel_dim = 28  # 28x28 imágenes
num_pixels = pixel_dim * pixel_dim
corner_pixels = [0, 27, 756, 783]  # Esquinas: arriba izq, arriba der, abajo izq, abajo der
n_atributos = 20  # cantidad fija de atributos a usar
valores_k = [1, 3, 5, 7, 9, 11]  # distintos k del modelo kNN


#%%

# Generamos los subconjuntos de pixeles usando diferentes criterios


# ordena por varianza en X_train
var_scores = X_train.var(axis=0).to_numpy()
sorted_by_var = np.argsort(var_scores)[::-1]  # Mayor a menor

# ordena los n atributos por varianza en X_train
var_scores = X_train.var(axis=0).to_numpy()
top_idx_var = np.argsort(var_scores)[-n_atributos:][::-1]

# ordena los atributos segun el criterio mutual info
mi_scores = mutual_info_classif(X_train, Y_train, discrete_features=False,  random_state=random_state)
top_idx_mi = np.argsort(mi_scores)[-n_atributos:][::-1]

# Cálculo para criterio aleatorio 
np.random.seed(random_state)
top_idx_rand = np.random.choice(num_pixels, size=n_atributos, replace=False)


# Función para elegir pixeles de las esquinas de manera rotativa
def get_rotating_corners(k):
    pattern = [0, 27, 783, 756]  # arriba izq, arriba der, abajo der, abajo izq
    return [pattern[i % 4] for i in range(k)]
# criterio para elegir pixeles de la esquina de manera rotativa
top_idx_corners = get_rotating_corners(n_atributos)


#%%

# inciso c. cantidad de k vecinos cte y varía cantidad de atributos usando diferentes subconjuntos de pixeles

# Inicializar resultados
results = {
    "varianza": [],
    "aleatorio": [],
    "esquinas": [],
    "mutual_info": []
}

# Comparar exactitud para cada k (cantidad de atributos)
for k in ks:
    # 1. Más variantes
    top_k_var = sorted_by_var[:k]
    X_train_var = X_train.iloc[:, top_k_var]
    X_test_var  = X_test.iloc[:, top_k_var]
    model_var   = KNeighborsClassifier(n_neighbors=3)
    model_var.fit(X_train_var, Y_train)
    results["varianza"].append(accuracy_score(Y_test, model_var.predict(X_test_var)))

    # 2. Aleatorios
    np.random.seed(random_state + k)
    rand_idx    = np.random.choice(num_pixels, size=k, replace=False)
    X_train_r   = X_train.iloc[:, rand_idx]
    X_test_r    = X_test.iloc[:, rand_idx]
    model_rand  = KNeighborsClassifier(n_neighbors=3)
    model_rand.fit(X_train_r, Y_train)
    results["aleatorio"].append(accuracy_score(Y_test, model_rand.predict(X_test_r)))

    # 3. Esquinas rotativas
    corner_idx  = get_rotating_corners(k)
    X_train_c   = X_train.iloc[:, corner_idx]
    X_test_c    = X_test.iloc[:, corner_idx]
    model_cr    = KNeighborsClassifier(n_neighbors=3)
    model_cr.fit(X_train_c, Y_train)
    results["esquinas"].append(accuracy_score(Y_test, model_cr.predict(X_test_c)))

    # 4. Mutual Info
    top_k_mi    = np.argsort(mi_scores)[-k:][::-1]
    X_train_mi  = X_train.iloc[:, top_k_mi]
    X_test_mi   = X_test.iloc[:, top_k_mi]
    model_mi    = KNeighborsClassifier(n_neighbors=3)
    model_mi.fit(X_train_mi, Y_train)
    results["mutual_info"].append(accuracy_score(Y_test, model_mi.predict(X_test_mi)))

# Graficar exactitud vs k
plt.figure(figsize=(10, 6))
for label, accs in results.items():
    plt.plot(ks, accs, marker='o', label=label.capitalize())
plt.xlabel("Cantidad de atributos (píxeles)")
plt.ylabel("Exactitud en test")
plt.title("kNN: Exactitud vs cantidad de pixeles \n 3 vecinos fijos sobre diferentes subconjuntos de pixeles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ————————————————————————————————————————————
# Ahora: encontrar el mejor caso y mostrar matriz de confusión + métricas
# ————————————————————————————————————————————

# 1) Localizar mejor criterio y k
best_acc = 0
best_crit = None
best_k = None
for crit, accs in results.items():
    for i, acc in enumerate(accs):
        if acc > best_acc:
            best_acc, best_crit, best_k = acc, crit, ks[i]

print(f"Mejor combinación → Criterio: {best_crit}, k_pix={best_k}, Exactitud={best_acc:.4f}")

# 2) Reconstruir subconjunto de píxeles
if best_crit == "varianza":
    best_idx = sorted_by_var[:best_k]
elif best_crit == "aleatorio":
    np.random.seed(random_state + best_k)
    best_idx = np.random.choice(num_pixels, size=best_k, replace=False)
elif best_crit == "esquinas":
    best_idx = get_rotating_corners(best_k)
else:  # mutual_info
    best_idx = np.argsort(mi_scores)[-best_k:][::-1]

# 3) Entrenar kNN con esa configuración y predecir
knn_best = KNeighborsClassifier(n_neighbors=3)
knn_best.fit(X_train.iloc[:, best_idx], Y_train)
y_pred_best = knn_best.predict(X_test.iloc[:, best_idx])

# 4) Mostrar matriz de confusión
cm = confusion_matrix(Y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=[0, 8])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Matriz Confusión de mejor criterio\n subcjto={best_crit.capitalize()}, k_pix={best_k}")
plt.show()

# 5) Mostrar classification report
print("Classification Report:")
print(classification_report(Y_test, y_pred_best, target_names=["Clase 0", "Clase 8"], zero_division=0))

#%%
# inciso d. cantidad de atributos cte y varía cantidad de k-vecinos en diferentes subconjuntos de pixeles.


# --- Diccionario con subconjuntos a comparar ---
subconjuntos = {
    "Varianza": top_idx_var,
    "Aleatorio": top_idx_rand,
    "Esquinas": top_idx_corners,
    "Mutual Info": top_idx_mi
}

# --- Evaluar cada subconjunto con distintos k (vecinos) ---
resultados = {nombre: [] for nombre in subconjuntos}

for nombre, indices in subconjuntos.items():
    X_train_sub = X_train.iloc[:, indices]
    X_test_sub = X_test.iloc[:, indices]
    
    for k_vecinos in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k_vecinos)
        modelo.fit(X_train_sub, Y_train)
        pred = modelo.predict(X_test_sub)
        acc = accuracy_score(Y_test, pred)
        resultados[nombre].append(acc)

# --- Graficar resultados ---
plt.figure(figsize=(10, 6))
for nombre, accuracies in resultados.items():
    plt.plot(valores_k, accuracies, marker='o', label=nombre)

plt.xlabel("Cantidad de vecinos (k)")
plt.ylabel("Exactitud en test")
plt.title(f"kNN: Exactitud vs k_vecinos \n {n_atributos} atributos fijos sobre diferentes subconjuntos de pixeles")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ————————————————————————————————————————————
# Post‑análisis: mejor criterio y mejor k_vecinos
# ————————————————————————————————————————————

# 1) Encontrar mejor combinación
best_acc   = 0
best_crit  = None
best_k     = None
for crit, accs in resultados.items():
    for i, acc in enumerate(accs):
        if acc > best_acc:
            best_acc, best_crit, best_k = acc, crit, valores_k[i]

print(f"\nMejor combinación → Criterio: {best_crit}, k_vecinos = {best_k}, Exactitud = {best_acc:.4f}\n")

# 2) Reconstruir índice de píxeles para ese criterio
if best_crit == "Varianza":
    best_idx = top_idx_var
elif best_crit == "Aleatorio":
    best_idx = top_idx_rand
elif best_crit == "Esquinas":
    best_idx = top_idx_corners
else:  # Mutual Info
    best_idx = top_idx_mi

# 3) Entrenar kNN con la mejor configuración y predecir
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train.iloc[:, best_idx], Y_train)
y_pred   = knn_best.predict(X_test.iloc[:, best_idx])

# 4) Mostrar matriz de confusión
cm   = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0, 8])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap="Blues", values_format="d", ax=ax)
ax.set_title(f"Matriz Confusión kNN de mejor criterio \n subcjto={best_crit}, k_vecinos={best_k}")
plt.show()

# 5) Mostrar classification report
print("Classification Report (mejor caso):")
print(classification_report(Y_test, y_pred, target_names=["Clase 0", "Clase 8"], zero_division=0))
#%%

# COMPARACION DE CLASIFICACION BINARIA CON MODELO KNN, TREE, REGRESION LOGISTICA

# ----------------------------------------------------
# 1) Definí el subconjunto de píxeles que quieras usar
# ----------------------------------------------------
k_pix_use = 50
idx_use   = top_idx_rand

X_tr = X_train.iloc[:, idx_use]
X_te = X_test .iloc[:, idx_use]

# ----------------------------------------------------
# 2) Configurá tus modelos con los hiperparámetros elegidos
# ----------------------------------------------------
models = {
    "kNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Tree (d=5)": DecisionTreeClassifier(
                       max_depth=5,
                       criterion="gini",
                       min_samples_leaf=1,
                       random_state=42
                   ),
    "LogReg": LogisticRegression(
                   solver="liblinear",
                   max_iter=300,
                   random_state=42
               )
}

# Vector binario para ROC AUC: 1=clase 8, 0=clase 0
y_test_bin = (Y_test == 8).astype(int)

# ----------------------------------------------------
# 3) Entrenar, predecir y comparar
# ----------------------------------------------------
for name, clf in models.items():
    # Entrenar
    clf.fit(X_tr, Y_train)
    # Predecir
    y_pred = clf.predict(X_te)

    # Métricas binarias: pos_label=8
    acc  = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred, pos_label=8)
    rec  = recall_score(   Y_test, y_pred, pos_label=8)
    f1   = f1_score(       Y_test, y_pred, pos_label=8)

    # ROC AUC
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_te)[:, 1]  # prob de clase 8
        auc     = roc_auc_score(y_test_bin, y_score)
    else:
        auc = float("nan")

    # Impresión resumida
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if not np.isnan(auc):
        print(f"ROC AUC  : {auc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(Y_test, y_pred, labels=[0, 8])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Clase 0", "Clase 8"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(f"{name} — Confusión")
    plt.tight_layout()
    plt.show()

    # Classification report completo
    print(classification_report(
        Y_test, y_pred,
        target_names=["Clase 0", "Clase 8"],
        zero_division=0,
        digits=4
    ))

#%%
#################################
##  CLASIFICACION MULTICLASE   ##
#################################

#%%
# Separar datos en features y labels
X = df.drop("label", axis=1)
Y = df["label"]

# 20% test held-out
X_dev, X_hld, Y_dev, Y_hld = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# 20% test dentro del 80% restante
X_train, X_test, Y_train, Y_test = train_test_split(
    X_dev, Y_dev, test_size=0.2, stratify=Y_dev, random_state=42
)

print("cantaidad de cada clase", df["label"].value_counts())
print(Y.shape,Y_hld.shape,Y_train.shape,Y_test.shape)
#%%
# EXACTITUD DE DIFERENTES MAX_DEPTH del arbol

# --- 1) (Opcional) Submuestreo para prototipar rápido en X_train/Y_train ---
speedup = False
if speedup:
    X_train_small, _, Y_train_small, _ = train_test_split(
        X_train, Y_train,
        train_size=0.3,
        stratify=Y_train,
        random_state=42
    )
else:
    X_train_small, Y_train_small = X_train, Y_train

# --- 2) Probar max_depth entre 1 y 10 usando X_train_small → X_test (dev dentro de X_dev) ---
depths = list(range(1, 11))
accuracies = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train_small, Y_train_small)
    
    # Aquí X_test/Y_test es tu conjunto “dev” (parte de X_dev)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    accuracies.append(acc)
    print(f"max_depth = {d:2d}  → accuracy_dev = {acc:.4f}")

# --- 3) Graficar accuracy vs max_depth ---
plt.figure(figsize=(8, 5))
plt.plot(depths, accuracies, marker='o')
plt.xticks(depths)
plt.xlabel("Profundidad máxima (max_depth)")
plt.ylabel("Exactitud en dev (X_test)")
plt.title("Decision Tree: exactitud vs max_depth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%

# EXACTITUD DE DIFERENTES MAX_DEPTH del arbol usando k-fold

# Supuestos: ya tenés X_dev, Y_dev definidos
# X_dev, Y_dev = ...

# 1) Preparar el rango de profundidades y el k-fold
depths = list(range(1, 11))
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2) Para cada max_depth, evaluamos con cross_val_score
cv_scores = []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(
        tree, X_dev, Y_dev,
        cv=kf,
        scoring="accuracy",
        n_jobs=-1
    )
    cv_scores.append(scores.mean())
    print(f"max_depth={d:2d} → CV accuracy mean = {scores.mean():.4f}  (std={scores.std():.4f})")

# 3) Elegir el mejor
best_idx = int(np.argmax(cv_scores))
best_depth = depths[best_idx]
best_score = cv_scores[best_idx]
print(f"\nMejor max_depth = {best_depth}, con accuracy CV = {best_score:.4f}")

# 4) (Opcional) Graficar accuracy vs max_depth
plt.figure(figsize=(8, 5))
plt.plot(depths, cv_scores, marker='o')
plt.xticks(depths)
plt.xlabel("Profundidad máxima (max_depth)")
plt.ylabel("Exactitud media (5-fold CV)")
plt.title("decision tree: exactitud vs max_depth (k_fold)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# GRID SEARCH de diferentes hiperparametros usando la profundidad de arbol con mejor exactitud (aclaracion demora un poco)

# 1) Definimos el clasificador con max_depth fijo que mejor resultado nos dió.
base_tree = DecisionTreeClassifier(
    max_depth=9,
    random_state=42
)

# 2) Armamos la grilla de diferentes hiperparametros
param_grid = {
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion":       ["gini", "entropy"],
    "splitter":        ["best", "random"]
}

# 3) Configuramos y ejecutamos el GridSearch
grid = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_dev, Y_dev)

print("Mejores hiperparámetros encontrados:")
print(grid.best_params_)
print(f"Mejor accuracy CV: {grid.best_score_:.4f}")

#%%
# GRAFICO DE HEATMAP DEL GRID SEARCH ANTERIOR


df = pd.DataFrame(grid.cv_results_)

# Elegimos los tres parámetros que queremos visualizar
param1 = "param_min_samples_leaf"
param2 = "param_splitter"
param3 = "param_criterion"
metric = "mean_test_score"

# Para cada valor de param3, dibujamos un heatmap de param1 vs param2
unique_p3 = df[param3].unique()

fig, axes = plt.subplots(
    nrows=1, ncols=len(unique_p3),
    figsize=(5 * len(unique_p3), 4),
    constrained_layout=True
)

for ax, val3 in zip(axes, unique_p3):
    sub = df[df[param3] == val3]
    # Pivot: filas = param1, columnas = param2, valores = mean_test_score
    pivot = sub.pivot(index=param1, columns=param2, values=metric)
    
    im = ax.imshow(pivot, origin='lower', aspect='auto')
    ax.set_title(f"{param3} = {val3}")
    ax.set_xlabel(param2)
    ax.set_ylabel(param1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Colorbar sólo en el primer subplot
    if ax is axes[0]:
        cbar = fig.colorbar(im, ax=axes.tolist(), orientation='vertical')
        cbar.set_label("Accuracy (CV mean)")

plt.suptitle("Heatmaps de CV accuracy para combinaciones de hiperparámetros", y=1.09)
plt.show()
#%%
# PREDECIMOS SOBRE EL HOLD-OUT CON LAS MEJORES CONFIGURACIONES Y OBTENEMOS LOS RESULTADOS
 
# 1) Definir y entrenar el modelo final de árbol con los hiperparámetros dados
tree_final = DecisionTreeClassifier(
    criterion="entropy",
    min_samples_leaf=1,
    splitter="best",
    random_state=42
)
tree_final.fit(X_dev, Y_dev)

# 2) Predecir sobre el hold-out
y_true = Y_hld
y_pred = tree_final.predict(X_hld)

# 3) Calcular métricas
acc = accuracy_score(y_true, y_pred)
print(f"Decision Tree (entropy, min_samples_leaf=1, splitter=best) — Accuracy en hold-out: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0, digits=4))

# 4) Matriz de confusión normalizada (recall por clase)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=range(10))
disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format=".2f")
ax.set_title("Decision Tree — Matriz de confusión normalizada")
plt.show()

#%%
#F1-score por clase en hold-out grafico

# Nombres reales de las clases (según Fashion MNIST)
class_names = [
    "0 (Remera)", "1 (Pantalón)", "2 (Pulóver)", "3 (Vestido)", "4 (Campera)",
    "5 (Sandalia)", "6 (Camisa)", "7 (Zapatilla)", "8 (Bolso)", "9 (Botín)"
]

# Obtener F1 por clase
_, _, f1s, _ = precision_recall_fscore_support(Y_hld, y_pred, zero_division=0)

# Ordenar de menor a mayor
indices_ordenados = np.argsort(f1s)
f1s_ordenados = f1s[indices_ordenados]
clases_ordenadas = [class_names[i] for i in indices_ordenados]

# Graficar
plt.figure(figsize=(10, 5))
plt.bar(clases_ordenadas, f1s_ordenados, color="skyblue")
plt.title("F1-score por clase en hold-out (ordenado)")
plt.xlabel("Clase")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

#%%
# heatmap de precision y recall grafico en hold-out

# Obtener precisión y recall por clase
precisiones, recalls, _, _ = precision_recall_fscore_support(Y_hld, y_pred, zero_division=0)
df_metrics = pd.DataFrame({
    "Clase": class_names,
    "Precisión": precisiones,
    "Recall": recalls
})

df_metrics = df_metrics.set_index("Clase")

# Graficar heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(df_metrics.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Valor"})
plt.title("Precisión y Recall por clase — Árbol de decisión (hold-out)")
plt.yticks(rotation=0)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
#%%
#EXTRA

# comparacion entre knn, tree y logistic regresion sobre subconjuntos de 
# pixeles elegidos por diferentes criterios de importancia
# (ESTE BLOQUE DEMORA UN POCO)


# --- Parámetros generales ---
k_pix = 75  # cantidad de píxeles a usar en cada subconjunto
random_state = 42

# --- Selección de atributos (subconjuntos) ---

# 1) Mayor varianza
var_scores = X_dev.var(axis=0).to_numpy()
top_idx_var = np.argsort(var_scores)[-k_pix:]

# 2) Aleatorio
np.random.seed(random_state)
top_idx_rand = np.random.choice(X_dev.shape[1], size=k_pix, replace=False)

# 3) Mutual Information
mi_scores = mutual_info_classif(X_dev, Y_dev, discrete_features=False)
top_idx_mi = np.argsort(mi_scores)[-k_pix:]

# Diccionario con todos los subconjuntos
subconjuntos = {
    "Varianza": top_idx_var,
    "Aleatorio": top_idx_rand,
    "Mutual Info": top_idx_mi
}

# --- Modelos a comparar ---
modelos = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=1,
        splitter="best",
        max_depth=9,
        random_state=42
    ),
    "kNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(
        solver="saga",
        max_iter=300,
        n_jobs=-1,
        random_state=42
    )
}

# --- Evaluación ---
resultados = []

for criterio, idx_cols in subconjuntos.items():
    X_train_sub = X_dev.iloc[:, idx_cols]
    X_test_sub  = X_hld.iloc[:, idx_cols]

    for nombre_modelo, modelo in modelos.items():
        modelo.fit(X_train_sub, Y_dev)
        pred = modelo.predict(X_test_sub)
        acc = accuracy_score(Y_hld, pred)
        resultados.append({
            "Modelo": nombre_modelo,
            "Criterio de selección": criterio,
            "Exactitud hold-out": round(acc, 4)
        })

# --- Mostrar resultados en tabla ordenada ---
df_resultados = pd.DataFrame(resultados)
print(df_resultados.pivot(index="Modelo", columns="Criterio de selección", values="Exactitud hold-out"))

#%%

# GRAFICO DE BARRAS DE LOS RESULTADOS ANTERIORES

# --- Reutilizamos df_resultados del paso anterior ---
pivot1 = df_resultados.pivot(index="Modelo", columns="Criterio de selección", values="Exactitud hold-out")

# Calcular el promedio de cada fila (modelo) y ordenarlo
pivot1_sorted = pivot1.loc[pivot1.mean(axis=1).sort_values().index]

pivot1_sorted.plot(kind="bar", figsize=(9, 5), colormap="Set2", edgecolor="black")
plt.title(f"Comparación de exactitud por modelo ({k_pix} píxeles)")
plt.ylabel("Exactitud en hold-out")
plt.ylim(0.5, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(title="Criterio")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

pivot2 = df_resultados.pivot(index="Criterio de selección", columns="Modelo", values="Exactitud hold-out")

# Ordenar por promedio de cada fila (criterio)
pivot2_sorted = pivot2.loc[pivot2.mean(axis=1).sort_values().index]

pivot2_sorted.plot(kind="bar", figsize=(9, 5), colormap="Pastel1", edgecolor="black")
plt.title(f"Comparación de modelos por criterio de selección ({k_pix} píxeles)")
plt.ylabel("Exactitud en hold-out")
plt.ylim(0.5, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(title="Modelo")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
