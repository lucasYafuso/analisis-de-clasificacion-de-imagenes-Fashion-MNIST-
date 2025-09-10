
---

# Clasificación de imágenes (Fashion-MNIST)

```markdown
# Clasificación de prendas con aprendizaje supervisado (Fashion-MNIST)

## 📌 Descripción
Este proyecto aborda la tarea de **clasificación de imágenes** usando el dataset **Fashion-MNIST**.  
Se probaron diferentes técnicas de **selección de atributos**, reducción de dimensionalidad y modelos de clasificación (kNN, árboles de decisión, regresión logística).

El objetivo fue **comparar modelos** y demostrar que con solo una **fracción de píxeles (30–60)** es posible lograr una precisión muy cercana a la de usar la imagen completa.

## 🛠️ Tecnologías
- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn (kNN, Decision Trees, Logistic Regression)
- Jupyter Notebook

## 📂 Estructura


## 📊 Resultados principales
- **Clasificación binaria (clases 0 “remera” vs 8 “bolso”):**  
  - kNN alcanzó **96.8% de exactitud** usando solo 60 píxeles seleccionados aleatoriamente.  
  - Árbol de decisión obtuvo el **mejor recall** (97.3%).  
  - Regresión logística quedó apenas por debajo (~95%).  

- **Clasificación multiclase (10 clases):**  
  - Árbol de decisión con `max_depth=9` logró **82% de exactitud** en el conjunto de validación.  
  - Las clases más fáciles (pantalones, bolsos, botas) superaron F1=0.9.  
  - La clase más difícil fue **camisa (F1≈0.54)**, confundida con remera y campera.  

![Matriz de confusión](ruta/a/imagen.png)

## 🌍 Relevancia
Este trabajo muestra cómo es posible **reducir drásticamente la dimensionalidad** en problemas de imágenes sin perder rendimiento, lo cual tiene impacto en eficiencia computacional.

## ▶️ Cómo usar
1. Clonar el repositorio.
2. Instalar dependencias:  
   ```bash
   pip install -r requirements.txt

