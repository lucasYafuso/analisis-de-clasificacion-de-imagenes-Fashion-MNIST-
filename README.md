# Clasificación de prendas con aprendizaje supervisado (Fashion-MNIST)

## Descripción
Este proyecto aborda la tarea de **clasificación de imágenes** usando el dataset **Fashion-MNIST**.  
Se probaron diferentes técnicas de **selección de atributos**, reducción de dimensionalidad y modelos de clasificación (kNN, árboles de decisión, regresión logística).

El objetivo fue **comparar modelos** y demostrar que con solo una **fracción de píxeles (30–60)** es posible lograr una precisión muy cercana a la de usar la imagen completa.

## Tecnologías
- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn (kNN, Decision Trees, Logistic Regression)
- Jupyter Notebook

## Estructura
- scrypt.py (todo el codigo separado en bloques)
- informe del analisis

## Resultados principales
- **Clasificación binaria (clases 0 “remera” vs 8 “bolso”):**  
  - kNN alcanzó **96.8% de exactitud** usando solo 60 píxeles seleccionados aleatoriamente.  
  - Árbol de decisión obtuvo el **mejor recall** (97.3%).  
  - Regresión logística quedó apenas por debajo (~95%).  

- **Clasificación multiclase (10 clases):**  
  - Árbol de decisión con `max_depth=9` logró **82% de exactitud** en el conjunto de validación.  
  - Las clases más fáciles (pantalones, bolsos, botas) superaron F1=0.9.  
  - La clase más difícil fue **camisa (F1≈0.54)**, confundida con remera y campera.  

## Relevancia
Este trabajo muestra cómo es posible **reducir drásticamente la dimensionalidad** en problemas de imágenes sin perder rendimiento, lo cual tiene impacto en eficiencia computacional.

## Cómo usar
1. Clonar el repositorio.
Podés instalar todas las dependencias con:

    pip install numpy pandas matplotlib seaborn scikit-lear
   
INSTRUCCIONES DE EJECUCIÓN

3. Asegurate de tener instalado Python 3 y las bibliotecas mencionadas.

4. Descargá el dataset "Fashion-MNIST" en formato CSV. Por defecto, el archivo espera el CSV en:

    C:/Users/ameli/Desktop/Facultad/LaboDatos/TP2/Fashion-MNIST.csv

   Cambiá esa ruta (línea 30 del .py) al path correspondiente en tu computadora si es necesario.

5. El script generará múltiples visualizaciones y métricas que se mostrarán automáticamente en pantalla.
	
6. Tener en cuenta que en Clasificacion Binaria y Clasificacion Multiclase hay variables con el mismo nombre. Asigna correctamente las variables 
   de cada seccion antes de correr sus funciones (correr los bloques de código en orden).


