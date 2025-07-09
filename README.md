# Modelos 2 Proyecto Final.

----

*Universidad de Antioquia*
<br/>

# 📂 Phishing-Detection-ML

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()  
[![Notebook](https://img.shields.io/badge/colab-ready-orange)]()
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)]()  
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)]()  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat-square)]()  
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🎥 Video de Demostración

Puedes ver la demostración del proyecto en el siguiente enlace:

[Ver en YouTube](https://www.youtube.com/watch?v=uMqmyF75rGs&feature=youtu.be)


## 🔍 Descripción
Este repositorio contiene todo lo necesario para desarrollar y comparar distintos modelos de Machine Learning destinados a la **detección de páginas de phishing**.  
Partimos de un dataset con **10 000 registros** (5 000 phishing, 5 000 legítimos) y **48 características** extraídas vía Selenium WebDriver.

### 🎯 Objetivos del Proyecto
- Desarrollar un sistema robusto de detección de phishing mediante técnicas de Machine Learning
- Comparar el rendimiento de diferentes algoritmos de clasificación
- Aplicar técnicas de reducción de dimensionalidad y selección de características
- Optimizar hiperparámetros para mejorar el rendimiento de los modelos
- Evaluar el impacto de diferentes técnicas de preprocesamiento en la precisión del modelo

### 🛠️ Metodología
El proyecto sigue una metodología estructurada que incluye:
1. **Análisis exploratorio de datos** - Comprensión del dataset y visualización de patrones
2. **Reducción de dimensionalidad** - Aplicación de técnicas como PCA y selección de características
3. **Modelado y optimización** - Implementación de múltiples algoritmos con optimización de hiperparámetros
4. **Evaluación comparativa** - Análisis de métricas de rendimiento y selección del mejor modelo

---
## 📄 Informe del Proyecto

Puedes consultar el informe técnico completo del proyecto en el siguiente enlace:

[📘 Informe_Phishing_For_ML.pdf](.Informe_Phishing_For_ML_Final.pdf)

---

## 📁 Estructura del Proyecto

### 📊 Análisis Exploratorio de Datos
- **`Exploring_Data.ipynb`** - Análisis inicial del dataset, visualización de distribuciones, correlaciones entre variables y patrones en los datos

### 🔍 Reducción de Dimensionalidad (`DimensionReduction/`)
Esta carpeta contiene los experimentos de reducción de dimensionalidad y selección de características:

- **`PCA_Feature_extraction.ipynb`** - Implementación del Análisis de Componentes Principales (PCA) para extracción de características
- **`Features_Selection_Filter.ipynb`** - Métodos de selección de características basados en filtros (correlación, chi-cuadrado, ANOVA)
- **`RF_&_SVM_Features_Selection_Filter.ipynb`** - Selección de características aplicada específicamente a Random Forest y SVM usando métodos de filtro
- **`Feature_Selection_RFECV_&_SFS.ipynb`** - Eliminación recursiva de características con validación cruzada (RFECV) y selección secuencial hacia adelante (SFS)
- **`RF_&_SVM_&_Features_Selection_Wrapper.ipynb`** - Métodos wrapper para selección de características en Random Forest y SVM

### 🤖 Modelos con Optimización de Hiperparámetros (`ModelsWithOptimizationHyperparameters/`)
Esta carpeta contiene la implementación y optimización de diferentes algoritmos de Machine Learning:

- **`KNN.ipynb`** - K-Nearest Neighbors con búsqueda de hiperparámetros óptimos
- **`LogisticRegression.ipynb`** - Regresión Logística con regularización y optimización de parámetros
- **`MLP.ipynb`** - Multi-Layer Perceptron (Red Neuronal) con optimización de arquitectura y parámetros
- **`RandomForest-Copy1.ipynb`** - Random Forest con optimización de número de árboles y profundidad
- **`SVM.ipynb`** - Support Vector Machine con optimización de kernel y parámetros

### 📄 Datos
- **`Phishing_Legitimate_full.csv`** - Dataset principal con 10,000 registros balanceados (5,000 phishing + 5,000 legítimos)

---

---

## 🧪 Técnicas y Algoritmos Implementados

### 📐 Reducción de Dimensionalidad
- **PCA (Principal Component Analysis)** - Extracción de componentes principales para reducir la dimensionalidad
- **Selección de Características por Filtros** - Chi-cuadrado, ANOVA F-test, correlación
- **Selección de Características Wrapper** - RFECV (Recursive Feature Elimination with Cross-Validation)
- **Selección Secuencial** - SFS (Sequential Forward Selection)

### 🤖 Algoritmos de Machine Learning
- **K-Nearest Neighbors (KNN)** - Clasificador basado en proximidad
- **Regresión Logística** - Modelo lineal con regularización L2
- **Support Vector Machine (SVM)** - Clasificador con kernels RBF, polinomial y lineal
- **Random Forest** - Ensemble de árboles de decisión
- **Multi-Layer Perceptron (MLP)** - Red neuronal multicapa

### 🎯 Técnicas de Optimización
- **Usamos un Framework de busqueda de hiperparámetros** llamado Optuna
- **Métricas de Evaluación** - Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📋 Estructura de las variables de nuestro dataset.

| No. | Feature                                      | Type          | Description                                                                                       |
|-----|----------------------------------------------|---------------|---------------------------------------------------------------------------------------------------|
| 1   | NumDots                                      | Discrete      | Counts the number of dots in webpage URL                                                           |
| 2   | SubdomainLevel                               | Discrete      | Counts the level of subdomain in webpage URL                                                      |
| 3   | PathLevel                                    | Discrete      | Counts the depth of the path in webpage URL                                                       |
| 4   | UrlLength                                    | Discrete      | Counts the total characters in the webpage URL                                                    |
| 5   | NumDash                                      | Discrete      | Counts the number of “-” in webpage URL                                                           |
| 6   | NumDashInHostname                            | Discrete      | Counts the number of “-” in hostname part of webpage URL                                          |
| 7   | AtSymbol                                     | Binary        | Checks if “@” symbol exist in webpage URL                                                         |
| 8   | TildeSymbol                                  | Binary        | Checks if “∼” symbol exist in webpage URL                                                         |
| 9   | NumUnderscore                                | Discrete      | Counts the number of “_” in webpage URL                                                           |
| 10  | NumPercent                                   | Discrete      | Counts the number of “%” in webpage URL                                                           |
| 11  | NumQueryComponents                           | Discrete      | Counts the number of query parts in webpage URL                                                   |
| 12  | NumAmpersand                                 | Discrete      | Counts the number of “&” in webpage URL                                                           |
| 13  | NumHash                                      | Discrete      | Counts the number of “#” in webpage URL                                                           |
| 14  | NumNumericChars                              | Discrete      | Counts the number of numeric characters in the webpage URL                                        |
| 15  | NoHttps                                      | Binary        | Checks if HTTPS exist in webpage URL                                                              |
| 16  | RandomString                                 | Binary        | Checks if random strings exist in webpage URL                                                     |
| 17  | IpAddress                                    | Binary        | Checks if IP address is used in hostname part of webpage URL                                      |
| 18  | DomainInSubdomains                           | Binary        | Checks if TLD or ccTLD is used as part of subdomain in webpage URL                               |
| 19  | DomainInPaths                                | Binary        | Checks if TLD or ccTLD is used in the path of webpage URL                                        |
| 20  | HttpsInHostname                              | Binary        | Checks if HTTPS is obfuscated in hostname part of webpage URL                                    |
| 21  | HostnameLength                               | Discrete      | Counts the total characters in hostname part of webpage URL                                       |
| 22  | PathLength                                   | Discrete      | Counts the total characters in path of webpage URL                                               |
| 23  | QueryLength                                  | Discrete      | Counts the total characters in query part of webpage URL                                         |
| 24  | DoubleSlashInPath                            | Binary        | Checks if “//” exist in the path of webpage URL                                                   |
| 25  | NumSensitiveWords                            | Discrete      | Counts the number of sensitive words (e.g., “secure,” “account,” “login,” etc.) in webpage URL   |
| 26  | EmbeddedBrandName                            | Binary        | Checks if brand name appears in subdomains and path; brand = most frequent domain in HTML        |
| 27  | PctExtHyperlinks                             | Continuous    | Percentage of external hyperlinks in HTML source code                                            |
| 28  | PctExtResourceUrls                           | Continuous    | Percentage of external resource URLs in HTML source code                                         |
| 29  | ExtFavicon                                   | Binary        | Checks if favicon is loaded from a different domain                                              |
| 30  | InsecureForms                                | Binary        | Checks if form action URL lacks HTTPS protocol                                                    |
| 31  | RelativeFormAction                           | Binary        | Checks if form action URL is relative                                                            |
| 32  | ExtFormAction                                | Binary        | Checks if form action URL is external domain                                                     |
| 33  | AbnormalFormAction                           | Categorical   | Form action contains “#,” “about:blank,” empty, or “javascript:true”                             |
| 34  | PctNullSelfRedirectHyperlinks                | Continuous    | Pct. of hyperlinks empty, self-redirect (“#”), current URL, or abnormal                          |
| 35  | FrequentDomainNameMismatch                   | Binary        | Most frequent domain in HTML ≠ webpage URL domain                                                |
| 36  | FakeLinkInStatusBar                          | Binary        | Checks if onMouseOver JS shows fake URL in status bar                                            |
| 37  | RightClickDisabled                           | Binary        | Checks if JS disables right-click function                                                       |
| 38  | PopUpWindow                                  | Binary        | Checks if JS launches pop-up windows                                                             |
| 39  | SubmitInfoToEmail                            | Binary        | Checks if HTML uses “mailto” in forms                                                            |
| 40  | IframeOrFrame                                | Binary        | Checks if iframe or frame tags are used                                                          |
| 41  | MissingTitle                                 | Binary        | Checks if the <title> tag is empty                                                               |
| 42  | ImagesOnlyInForm                             | Binary        | Form scope contains only images, no text                                                         |
| 43  | SubdomainLevelRT                             | Categorical   | Counts dots in hostname (reinforced thresholds)                                                  |
| 44  | UrlLengthRT                                  | Categorical   | URL length with applied rules/thresholds                                                         |
| 45  | PctExtResourceUrlsRT                         | Categorical   | Pct. of external resource URLs (categorical)                                                     |
| 46  | AbnormalExtFormActionR                       | Categorical   | Form action foreign domain, “about:blank”, or empty                                              |
| 47  | ExtMetaScriptLinkRT                          | Categorical   | Pct. of meta, script, link tags with external URLs                                               |
| 48  | PctExtNullSelfRedirect-HyperlinksRT          | Categorical   | Pct. of hyperlinks external, “#,” or “JavaScript:: void(0)”                                      |

---


