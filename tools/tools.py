import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def preprocesamiento(dataset):
    """
    Preprocesses exoplanet datasets for machine learning analysis.
    This function loads a specified exoplanet dataset (KOI, K2, or TOI), performs 
    data cleaning and preprocessing steps, and prepares the data for machine learning tasks.
    Parameters
    ----------
    dataset : str
        The name of the dataset to preprocess ('koi', 'k2', or 'toi').
        The function expects the corresponding CSV file to be in the '../data/' directory.
    Returns
    -------
    X_train_scaled : numpy.ndarray
        Scaled training feature data for confirmed/false positive exoplanets.
    X_test_scaled : numpy.ndarray
        Scaled test feature data for confirmed/false positive exoplanets.
    y_train : numpy.ndarray
        Encoded target values (CONFIRMED=1, FALSE POSITIVE=0) for training data.
    y_test : numpy.ndarray
        Encoded target values (CONFIRMED=1, FALSE POSITIVE=0) for test data.
    X_candidate_scaled : numpy.ndarray
        Scaled feature data for candidate exoplanets.
    """
    path = "../data/" + dataset + ".csv"
    df = pd.read_csv(path, comment = '#')
    df.drop(columns=["rowid"], inplace=True)
    if dataset == 'koi':
        df.rename(columns={'koi_disposition': 'disposition'}, inplace=True)
        df_confirmed = df[df["disposition"].isin(["FALSE POSITIVE", "CONFIRMED"])].copy()
        df_candidate = df[df["disposition"].isin(["CANDIDATE"])].copy()
    elif dataset == 'k2':
        df_confirmed = df[df["disposition"].isin(["FALSE POSITIVE", "CONFIRMED"])].copy()
        df_candidate = df[df["disposition"].isin(["CANDIDATE"])].copy()
    elif dataset == 'toi':
        df.rename(columns={'tfopwg_disp': 'disposition'}, inplace=True)
        df = df[df['disposition'].isin(["CP", "KP", "FP", "PC"])].copy()
        disposition_mapping = {'CP': 'CONFIRMED', 'KP': 'CONFIRMED', 'FP': 'FALSE POSITIVE', 'PC': 'CANDIDATE'}
        df['disposition'] = df['disposition'].map(disposition_mapping)
        df_confirmed = df[df["disposition"].isin(["FALSE POSITIVE", "CONFIRMED"])].copy()
        df_candidate = df[df["disposition"].isin(["CANDIDATE"])].copy()
        
    # Borrar columnas que los candidatos tengan vacías
    for column in df_candidate.columns:
        if df_candidate[column].nunique() == 0:
            df_confirmed.drop(columns= column, inplace = True)
            df_candidate.drop(columns= column, inplace = True)
            
    # Borrar columnas que no son datos numéricos
    object_cols = df_candidate.select_dtypes(include=['object']).columns
    if 'disposition' in object_cols:
        object_cols = object_cols.drop('disposition')
    df_confirmed.drop(columns = object_cols, inplace = True)
    df_candidate.drop(columns = object_cols, inplace = True)

    # Eliminar columnas con pocos datos
    for column in df_candidate.columns:
        if df_candidate[column].notna().sum() < len(df_candidate)*0.9:
            df_candidate.drop(columns=column, inplace= True)
            df_confirmed.drop(columns=column, inplace= True)

    # Borrar filas que tengan al menos un dato nulo
    df_confirmed.dropna(inplace=True)
    df_candidate.dropna(inplace=True)

    # Borrar columnas con datos iguales
    for column in df_candidate.columns:
        if df_candidate[column].nunique() == 1:
            if column != 'disposition':
                df_candidate.drop(columns=[column])
                df_confirmed.drop(columns=[column])
 
    # Separar datos en X y y
    y_confirmed = df_confirmed["disposition"]
    X_confirmed = df_confirmed.drop(columns = ["disposition"])
    X_candidate = df_candidate.drop(columns = ["disposition"])

    # Categorizar variable y usando map
    y_confirmed_encoded = y_confirmed.map({"FALSE POSITIVE": 0, "CONFIRMED": 1}).values

    # Separar en train y test los datos confirmados
    X_train, X_test, y_train, y_test = train_test_split(X_confirmed, y_confirmed_encoded, test_size=0.2, random_state=42, stratify=y_confirmed_encoded)

    # Estandarizar datos
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    X_candidate_scaled = pd.DataFrame(
        scaler.transform(X_candidate),
        columns=X_candidate.columns,
        index=X_candidate.index
    )


    return X_train_scaled, X_test_scaled, y_train, y_test, X_candidate_scaled


def calcular_metricas(model, X, y):
    """
    Esta función calcula las métricas de desempeño del modelo, ya sea para train o test

    Parámetros:
    - model (objeto de modelo): Modelo entrenado
    - X (array-like): Conjunto de características (entradas)
    - y (array-like): Valores reales (etiquetas)

    Retorna:
    - CM (ndarray): Matriz de confusión (valores reales vs predichos)
    - accuracy (float): Exactitud global del modelo
    - precision (float): Precisión en la clasificación de exoplanetas confirmados
    - recall (float): Sensibilidad o tasa de verdaderos positivos
    - f1 (float): Media armónica entre precisión y recall.
    - roc_auc (float): Área bajo la curva ROC, que mide la capacidad de discriminación del modelo
    """
    #Probabilidades predichas
    y_prob = model.predict(X, verbose=0).flatten()

    #Predicción de clase
    y_pred = (y_prob > 0.5).astype(int)
    y_true = y.flatten() if y.ndim > 1 else y

    CM = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    return CM, accuracy, precision, recall, f1, roc_auc


def evaluar_modelo(model, X_train, y_train, X_test, y_test, dataset):
    """
    Esta función evalúa el desempeño del modelo en train y test, generando las matrices de confusión junto con las métricas de desempeño

    Parámetros:
    - model (objeto de modelo): Modelo entrenado
    - X_train (array-like): Conjunto de características de entrenamiento
    - y_train (array-like): Etiquetas de entrenamiento
    - X_test (array-like): Conjunto de características de prueba
    - y_test (array-like): Etiquetas de prueba
    - dataset (str): Nombre del conjunto de datos
    """
    cm_train, acc_train, prec_train, rec_train, f1_train, roc_train = calcular_metricas(model, X_train, y_train)
    cm_test, acc_test, prec_test, rec_test, f1_test, roc_test = calcular_metricas(model, X_test, y_test)

    clases = ['FALSE POSITIVE', 'CONFIRMED']

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(cm_train, annot=True, fmt='d', cmap='BuPu',
                xticklabels=clases, yticklabels=clases,
                ax=ax[0], annot_kws={'size': 10})
    ax[0].set_title(f'{dataset} - Train', fontsize=14)
    ax[0].set_xlabel('Predicho', fontsize=12)
    ax[0].set_ylabel('Real', fontsize=12)
    ax[0].tick_params(labelsize=11)
    ax[0].text(0.45, -0.12, f'\nAccuracy: {acc_train:.4f} | Precisión: {prec_train:.4f}\nRecall: {rec_train:.4f} | F1-Score: {f1_train:.4f}\nROC-AUC: {roc_train:.4f}',
            transform=ax[0].transAxes, ha='center', fontsize=11, va='top')

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='BuPu',
                xticklabels=clases, yticklabels=clases,
                ax=ax[1], annot_kws={'size': 10})
    ax[1].set_title(f'{dataset} - Test', fontsize=14)
    ax[1].set_xlabel('Predicho', fontsize=12)
    ax[1].set_ylabel('Real', fontsize=12)
    ax[1].tick_params(labelsize=11)
    ax[1].text(0.45, -0.12, f'\nAccuracy: {acc_test:.4f} | Precisión: {prec_test:.4f}\nRecall: {rec_test:.4f} | F1-Score: {f1_test:.4f}\nROC-AUC: {roc_test:.4f}',
            transform=ax[1].transAxes, ha='center', fontsize=11, va='top')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25) 
    plt.savefig(f'CM_{dataset}.png', dpi=300, bbox_inches='tight')
    plt.show()


def predecir_exoplanetas(model, X):
    """
    Esta función realiza predicciones con el modelo entrenado sobre un conjunto de datos,
    devolviendo las probabilidades de que un objeto sea un exoplaneta confirmado o un falso positivo

    Parámetros:
    - model (objeto de modelo): Modelo entrenado
    - X (array-like): Conjunto de características sobre las que se realizarán las predicciones

    Retorna:
    - prob_fp (ndarray): Probabilidad de que el objeto sea un falso positivo
    - prob_confirmed (ndarray): Probabilidad estimada de que el objeto sea un exoplaneta
    - predicciones (ndarray de str): Clases predichas en texto ("FALSE POSITIVE" o "CONFIRMED") para cada muestra
    """
    prob_confirmed = model.predict(X).flatten() #Probabilidad de ser CONFIRMED
    prob_fp = 1 - prob_confirmed #Probabilidad de ser FALSE POSITIVE
    
    #Predicciones
    preds = (prob_confirmed >= 0.5).astype(int)
    clases = {0: 'FALSE POSITIVE', 1: 'CONFIRMED'}
    predicciones = np.array([clases[p] for p in preds])

    return prob_fp, prob_confirmed, predicciones