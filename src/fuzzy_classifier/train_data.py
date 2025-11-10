import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

absolute_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lunar_game_data = pd.read_csv(os.path.join(absolute_root_path, 'data/sheets/LunarLanderv2_Obs.csv'))

def getTrainedLanderData(data=lunar_game_data):
    """
    Carga los datos del juego LunarLander y devuelve un DataFrame con los datos de entrenamiento.
    Los datos se normalizan y se dividen en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba).
    """
    # Obtener subconjunto X & subconjunto y

    columnas = ['X Coordinate', 'Y Coordinate', 'Linear Velocity X', 'Linear Velocity Y']

    # * Nota: se selecciona desde el episodio 3950 porque es el episodio en el que el agente comienza a aprender a aterrizar
    trained_lander_data = data.loc[data['Episode No'] >= 3950]

    lander_data = trained_lander_data[columnas]

    action_data = trained_lander_data[['Action']]

    # Normalizar datos

    lander_data = lander_data.apply(lambda x: x / np.max(x))

    # Entrenar conjunto de datos (datos de prueba 20%)

    X_train, X_test, y_train, y_test = train_test_split(lander_data, action_data, test_size=0.2, random_state=0)

    # Reindexar todos
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def getClassArray(classNumber, X_train, y_train):
    """
    Devuelve array conteniendo todas las instancias de la clasenumero del X_train
    """
    listOfInstancesIndex = y_train.index[y_train['Action'] == classNumber].tolist()
    # print(listOfInstancesIndex)

    return X_train.iloc[listOfInstancesIndex, :]

