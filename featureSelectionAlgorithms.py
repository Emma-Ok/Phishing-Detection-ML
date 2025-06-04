import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

class FeatureSelectionAlgorithms:
    """
    Clase que implementa algoritmos de selección y extracción de características
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features = None
        
    def lasso_feature_selection(self, X, y, alpha=None, plot=True):
        """
        Selección de características usando LASSO
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Matriz de características
        y : array-like, shape (n_samples,)
            Vector objetivo
        alpha : float, opcional
            Parámetro de regularización. Si None, se usa validación cruzada
        plot : bool
            Si mostrar gráficos de los coeficientes
        
        Returns:
        --------
        X_selected : array-like
            Características seleccionadas
        selected_indices : array
            Índices de las características seleccionadas
        """
        print("=== SELECCIÓN DE CARACTERÍSTICAS CON LASSO ===")
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)
        
        if alpha is None:
            # Usar validación cruzada para encontrar el mejor alpha
            lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
            lasso_cv.fit(X_scaled, y)
            alpha = lasso_cv.alpha_
            print(f"Alpha óptimo encontrado: {alpha:.4f}")
        
        # Aplicar LASSO
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        # Obtener características seleccionadas (coeficientes != 0)
        selected_indices = np.where(lasso.coef_ != 0)[0]
        X_selected = X_scaled[:, selected_indices]
        
        print(f"Características originales: {X.shape[1]}")
        print(f"Características seleccionadas: {len(selected_indices)}")
        print(f"Índices seleccionados: {selected_indices}")
        
        if plot and len(selected_indices) > 0:
            plt.figure(figsize=(12, 6))
            
            # Gráfico de coeficientes
            plt.subplot(1, 2, 1)
            plt.bar(range(len(lasso.coef_)), lasso.coef_)
            plt.title('Coeficientes LASSO')
            plt.xlabel('Índice de característica')
            plt.ylabel('Coeficiente')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            
            # Path de regularización
            plt.subplot(1, 2, 2)
            alphas = np.logspace(-4, 1, 50)
            coefs = []
            for a in alphas:
                lasso_temp = Lasso(alpha=a, max_iter=2000)
                lasso_temp.fit(X_scaled, y)
                coefs.append(lasso_temp.coef_)
            
            coefs = np.array(coefs)
            for i in range(coefs.shape[1]):
                plt.plot(alphas, coefs[:, i], alpha=0.7)
            plt.xscale('log')
            plt.xlabel('Alpha')
            plt.ylabel('Coeficientes')
            plt.title('Path de Regularización LASSO')
            plt.axvline(x=alpha, color='r', linestyle='--', label=f'Alpha óptimo: {alpha:.4f}')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        self.selected_features = selected_indices
        return X_selected, selected_indices
    
    def pca_feature_extraction(self, X, n_components=None, variance_threshold=0.95, plot=True):
        """
        Extracción de características usando PCA
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Matriz de características
        n_components : int, opcional
            Número de componentes. Si None, se determina por variance_threshold
        variance_threshold : float
            Umbral de varianza explicada acumulada
        plot : bool
            Si mostrar gráficos de PCA
        
        Returns:
        --------
        X_pca : array-like
            Datos transformados por PCA
        pca : PCA object
            Objeto PCA ajustado
        """
        print("\n=== EXTRACCIÓN DE CARACTERÍSTICAS CON PCA ===")
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)
        
        # Determinar número de componentes
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"Características originales: {X.shape[1]}")
        print(f"Componentes principales: {n_components}")
        print(f"Varianza explicada por componente: {pca.explained_variance_ratio_}")
        print(f"Varianza explicada total: {pca.explained_variance_ratio_.sum():.4f}")
        
        if plot:
            plt.figure(figsize=(15, 10))
            
            # Varianza explicada por componente
            plt.subplot(2, 3, 1)
            plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_)
            plt.title('Varianza Explicada por Componente')
            plt.xlabel('Componente Principal')
            plt.ylabel('Proporción de Varianza')
            
            # Varianza explicada acumulada
            plt.subplot(2, 3, 2)
            plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    np.cumsum(pca.explained_variance_ratio_), 'bo-')
            plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                       label=f'Umbral: {variance_threshold}')
            plt.title('Varianza Explicada Acumulada')
            plt.xlabel('Número de Componentes')
            plt.ylabel('Varianza Acumulada')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Heatmap de componentes principales (si hay suficientes features)
            if X.shape[1] <= 20:
                plt.subplot(2, 3, 3)
                sns.heatmap(pca.components_, cmap='RdBu_r', center=0, 
                           xticklabels=range(X.shape[1]),
                           yticklabels=[f'PC{i+1}' for i in range(n_components)])
                plt.title('Componentes Principales\n(Pesos de características)')
                plt.xlabel('Característica Original')
                plt.ylabel('Componente Principal')
            
            # Proyección en las primeras dos componentes (si es posible)
            if n_components >= 2:
                plt.subplot(2, 3, 4)
                plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
                plt.title('Proyección en PC1 vs PC2')
                plt.grid(True, alpha=0.3)
            
            # Biplot (si hay pocas características)
            if X.shape[1] <= 10 and n_components >= 2:
                plt.subplot(2, 3, 5)
                # Puntos de datos
                plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
                
                # Vectores de características
                feature_vectors = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
                for i, (x, y) in enumerate(feature_vectors):
                    plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, 
                             fc='red', ec='red', alpha=0.7)
                    plt.text(x*1.1, y*1.1, f'F{i}', fontsize=10, ha='center', va='center')
                
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
                plt.title('Biplot PCA')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return X_pca, pca
    
    def lda_feature_extraction(self, X, y, plot=True):
        """
        Extracción de características usando LDA (Linear Discriminant Analysis)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Matriz de características
        y : array-like, shape (n_samples,)
            Vector de etiquetas
        plot : bool
            Si mostrar gráficos de LDA
        
        Returns:
        --------
        X_lda : array-like
            Datos transformados por LDA
        lda : LDA object
            Objeto LDA ajustado
        """
        print("\n=== EXTRACCIÓN DE CARACTERÍSTICAS CON LDA ===")
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)
        
        # Determinar número máximo de componentes
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        max_components = min(n_classes - 1, n_features)
        
        # Aplicar LDA
        lda = LDA(n_components=max_components)
        X_lda = lda.fit_transform(X_scaled, y)
        
        print(f"Características originales: {X.shape[1]}")
        print(f"Número de clases: {n_classes}")
        print(f"Componentes LDA: {X_lda.shape[1]}")
        print(f"Proporción de varianza explicada: {lda.explained_variance_ratio_}")
        
        if plot:
            plt.figure(figsize=(15, 5))
            
            # Varianza explicada por componente
            plt.subplot(1, 3, 1)
            components = range(1, len(lda.explained_variance_ratio_) + 1)
            plt.bar(components, lda.explained_variance_ratio_)
            plt.title('Varianza Explicada por Componente LDA')
            plt.xlabel('Componente Discriminante')
            plt.ylabel('Proporción de Varianza')
            
            # Proyección en el primer componente (si existe)
            if X_lda.shape[1] >= 1:
                plt.subplot(1, 3, 2)
                for class_label in np.unique(y):
                    mask = y == class_label
                    plt.hist(X_lda[mask, 0], alpha=0.7, label=f'Clase {class_label}')
                plt.xlabel('LD1')
                plt.ylabel('Frecuencia')
                plt.title('Distribución en LD1')
                plt.legend()
            
            # Proyección en las primeras dos componentes (si existen)
            if X_lda.shape[1] >= 2:
                plt.subplot(1, 3, 3)
                for class_label in np.unique(y):
                    mask = y == class_label
                    plt.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                              label=f'Clase {class_label}', alpha=0.7)
                plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.3f})')
                plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.3f})')
                plt.title('Proyección en LD1 vs LD2')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return X_lda, lda
    
    def fisher_discriminant(self, X, y, plot=True):
        """
        Implementación del discriminante de Fisher
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Matriz de características
        y : array-like, shape (n_samples,)
            Vector de etiquetas (debe ser binario)
        plot : bool
            Si mostrar gráficos
        
        Returns:
        --------
        w : array
            Vector de pesos del discriminante de Fisher
        X_fisher : array
            Proyección de los datos en la dirección de Fisher
        """
        print("\n=== DISCRIMINANTE DE FISHER ===")
        
        # Verificar que sea un problema de clasificación binaria
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("El discriminante de Fisher está implementado solo para clasificación binaria")
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X)
        
        # Separar clases
        class1_mask = y == classes[0]
        class2_mask = y == classes[1]
        
        X1 = X_scaled[class1_mask]
        X2 = X_scaled[class2_mask]
        
        # Calcular medias de cada clase
        mu1 = np.mean(X1, axis=0)
        mu2 = np.mean(X2, axis=0)
        
        # Calcular matrices de covarianza intra-clase
        S1 = np.cov(X1.T)
        S2 = np.cov(X2.T)
        
        # Matriz de covarianza intra-clase total
        Sw = S1 + S2
        
        # Vector de diferencia de medias
        mean_diff = mu2 - mu1
        
        # Calcular el vector de pesos de Fisher
        try:
            w = np.linalg.solve(Sw, mean_diff)
        except np.linalg.LinAlgError:
            # Si Sw no es invertible, usar pseudoinversa
            w = np.linalg.pinv(Sw) @ mean_diff
        
        # Normalizar el vector de pesos
        w = w / np.linalg.norm(w)
        
        # Proyectar los datos
        X_fisher = X_scaled @ w
        
        # Calcular métricas de separabilidad
        proj1 = X_fisher[class1_mask]
        proj2 = X_fisher[class2_mask]
        
        between_class_var = (np.mean(proj1) - np.mean(proj2))**2
        within_class_var = np.var(proj1) + np.var(proj2)
        fisher_ratio = between_class_var / within_class_var if within_class_var > 0 else 0
        
        print(f"Clases: {classes}")
        print(f"Tamaño clase {classes[0]}: {len(X1)}")
        print(f"Tamaño clase {classes[1]}: {len(X2)}")
        print(f"Ratio de Fisher: {fisher_ratio:.4f}")
        print(f"Vector de pesos shape: {w.shape}")
        
        if plot:
            plt.figure(figsize=(15, 5))
            
            # Histograma de proyecciones
            plt.subplot(1, 3, 1)
            plt.hist(proj1, alpha=0.7, bins=20, label=f'Clase {classes[0]}')
            plt.hist(proj2, alpha=0.7, bins=20, label=f'Clase {classes[1]}')
            plt.xlabel('Proyección de Fisher')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Proyecciones')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Vector de pesos
            plt.subplot(1, 3, 2)
            plt.bar(range(len(w)), w)
            plt.title('Vector de Pesos de Fisher')
            plt.xlabel('Índice de Característica')
            plt.ylabel('Peso')
            plt.grid(True, alpha=0.3)
            
            # Proyección 2D (si hay al menos 2 características)
            if X.shape[1] >= 2:
                plt.subplot(1, 3, 3)
                plt.scatter(X_scaled[class1_mask, 0], X_scaled[class1_mask, 1], 
                           alpha=0.7, label=f'Clase {classes[0]}')
                plt.scatter(X_scaled[class2_mask, 0], X_scaled[class2_mask, 1], 
                           alpha=0.7, label=f'Clase {classes[1]}')
                
                # Dibujar la dirección de Fisher
                center = np.mean(X_scaled, axis=0)
                direction = w[:2] if len(w) >= 2 else np.array([w[0], 0])
                plt.arrow(center[0], center[1], direction[0], direction[1], 
                         head_width=0.1, head_length=0.1, fc='red', ec='red', 
                         linewidth=2, label='Dirección Fisher')
                
                plt.xlabel('Característica 1')
                plt.ylabel('Característica 2')
                plt.title('Datos y Dirección de Fisher')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return w, X_fisher
    
    def compare_methods(self, X, y, test_size=0.3):
        """
        Comparar todos los métodos en términos de rendimiento de clasificación
        """
        print("\n" + "="*50)
        print("COMPARACIÓN DE MÉTODOS")
        print("="*50)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                           random_state=42, stratify=y)
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Datos originales
        from sklearn.linear_model import LogisticRegression
        clf_original = LogisticRegression(random_state=42, max_iter=1000)
        clf_original.fit(X_train_scaled, y_train)
        y_pred_original = clf_original.predict(X_test_scaled)
        results['Original'] = accuracy_score(y_test, y_pred_original)
        
        # LASSO
        try:
            X_train_lasso, selected_indices = self.lasso_feature_selection(X_train, y_train, plot=False)
            if len(selected_indices) > 0:
                X_test_lasso = self.scaler.transform(X_test)[:, selected_indices]
                clf_lasso = LogisticRegression(random_state=42, max_iter=1000)
                clf_lasso.fit(X_train_lasso, y_train)
                y_pred_lasso = clf_lasso.predict(X_test_lasso)
                results['LASSO'] = accuracy_score(y_test, y_pred_lasso)
            else:
                results['LASSO'] = 0.0
        except Exception as e:
            print(f"Error en LASSO: {e}")
            results['LASSO'] = 0.0
        
        # PCA
        try:
            X_train_pca, pca = self.pca_feature_extraction(X_train, plot=False)
            X_test_pca = pca.transform(self.scaler.transform(X_test))
            clf_pca = LogisticRegression(random_state=42, max_iter=1000)
            clf_pca.fit(X_train_pca, y_train)
            y_pred_pca = clf_pca.predict(X_test_pca)
            results['PCA'] = accuracy_score(y_test, y_pred_pca)
        except Exception as e:
            print(f"Error en PCA: {e}")
            results['PCA'] = 0.0
        
        # LDA
        try:
            X_train_lda, lda = self.lda_feature_extraction(X_train, y_train, plot=False)
            X_test_lda = lda.transform(self.scaler.transform(X_test))
            clf_lda = LogisticRegression(random_state=42, max_iter=1000)
            clf_lda.fit(X_train_lda, y_train)
            y_pred_lda = clf_lda.predict(X_test_lda)
            results['LDA'] = accuracy_score(y_test, y_pred_lda)
        except Exception as e:
            print(f"Error en LDA: {e}")
            results['LDA'] = 0.0
        
        # Fisher (solo para clasificación binaria)
        if len(np.unique(y)) == 2:
            try:
                w, _ = self.fisher_discriminant(X_train, y_train, plot=False)
                X_train_fisher = (self.scaler.fit_transform(X_train) @ w).reshape(-1, 1)
                X_test_fisher = (self.scaler.transform(X_test) @ w).reshape(-1, 1)
                clf_fisher = LogisticRegression(random_state=42, max_iter=1000)
                clf_fisher.fit(X_train_fisher, y_train)
                y_pred_fisher = clf_fisher.predict(X_test_fisher)
                results['Fisher'] = accuracy_score(y_test, y_pred_fisher)
            except Exception as e:
                print(f"Error en Fisher: {e}")
                results['Fisher'] = 0.0
        
        # Mostrar resultados
        print("\nACCURACY DE CLASIFICACIÓN:")
        print("-" * 30)
        for method, accuracy in results.items():
            print(f"{method:12}: {accuracy:.4f}")
        
        # Gráfico de comparación
        plt.figure(figsize=(10, 6))
        methods = list(results.keys())
        accuracies = list(results.values())
        
        bars = plt.bar(methods, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 
                                                  'gold', 'plum'][:len(methods)])
        plt.title('Comparación de Accuracy por Método')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return results

# Ejemplo de uso con diferentes datasets
def demo_feature_selection():
    """
    Demostración de los algoritmos con diferentes datasets
    """
    print("DEMOSTRACIÓN DE ALGORITMOS DE SELECCIÓN Y EXTRACCIÓN DE CARACTERÍSTICAS")
    print("="*70)
    
    # Crear instancia
    fs = FeatureSelectionAlgorithms()
    
    # Dataset 1: Sintético con características ruidosas
    print("\n" + "="*50)
    print("DATASET 1: DATOS SINTÉTICOS")
    print("="*50)
    
    X_synth, y_synth = make_classification(n_samples=500, n_features=20, 
                                          n_informative=5, n_redundant=5,
                                          n_clusters_per_class=1, random_state=42)
    
    print(f"Shape de datos sintéticos: {X_synth.shape}")
    print(f"Clases únicas: {np.unique(y_synth)}")
    
    # Aplicar todos los métodos
    fs.lasso_feature_selection(X_synth, y_synth)
    fs.pca_feature_extraction(X_synth)
    fs.lda_feature_extraction(X_synth, y_synth)
    fs.fisher_discriminant(X_synth, y_synth)
    fs.compare_methods(X_synth, y_synth)
    
    # Dataset 2: Iris (clásico)
    print("\n" + "="*50)
    print("DATASET 2: IRIS")
    print("="*50)
    
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    print(f"Shape de datos Iris: {X_iris.shape}")
    print(f"Características: {iris.feature_names}")
    print(f"Clases: {iris.target_names}")
    
    # Crear nueva instancia para iris
    fs_iris = FeatureSelectionAlgorithms()
    
    fs_iris.lasso_feature_selection(X_iris, y_iris)
    fs_iris.pca_feature_extraction(X_iris)
    fs_iris.lda_feature_extraction(X_iris, y_iris)
    fs_iris.compare_methods(X_iris, y_iris)
    
    # Dataset 3: Wine (para Fisher con clases binarias)
    print("\n" + "="*50)
    print("DATASET 3: WINE (BINARIO)")
    print("="*50)
    
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    
    # Convertir a binario (clase 0 vs resto)
    y_wine_binary = (y_wine == 0).astype(int)
    
    print(f"Shape de datos Wine: {X_wine.shape}")
    print(f"Clases binarias: {np.unique(y_wine_binary)} (0 vs resto)")
    
    # Crear nueva instancia para wine
    fs_wine = FeatureSelectionAlgorithms()
    
    fs_wine.lasso_feature_selection(X_wine, y_wine_binary)
    fs_wine.pca_feature_extraction(X_wine)
    fs_wine.lda_feature_extraction(X_wine, y_wine_binary)
    fs_wine.fisher_discriminant(X_wine, y_wine_binary)
    fs_wine.compare_methods(X_wine, y_wine_binary)

if __name__ == "__main__":
    # Ejecutar demostración
    demo_feature_selection()