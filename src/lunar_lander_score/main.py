import sys
import numpy as np
from random import randint

# setting path
sys.path.append('../')

from fuzzy_classifier.train_data import getTrainedLanderData
from fuzzy_classifier.generate_rule_classifier import (
    generateRules,
    getCompetitionStrength,
    N_INDIV,
    N_VARS,
    N_CLASSES,
    triangle_set,
)

if __name__ == "__main__":
    # print("variable triangulo", triangle_set)

    # Entrenamiento de datos
    X_train, X_test, y_train, y_test = getTrainedLanderData()


    rule_class = generateRules(N_INDIV, N_VARS)
    print(rule_class, type(rule_class))
    print(X_test.loc[0], type(X_test.loc[0]))
    # print(triangle_set.sets["medio"].membership(300))
    # array_mt = getMutationArray(rule_class, X_test.loc[0], triangle_set)
    array_mt = getCompetitionStrength(rule_class, N_CLASSES, X_train, y_train)
    print("competition strength", array_mt)
