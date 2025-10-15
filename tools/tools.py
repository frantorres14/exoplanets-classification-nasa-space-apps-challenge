import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

