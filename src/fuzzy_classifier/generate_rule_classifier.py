import numpy as np
from random import randint
# from inferfuzzy.systems import LarsenSystem, MamdaniSystem
from inferfuzzy.memberships import (
    LMembership,
)
from inferfuzzy import var
from sklearn.metrics import accuracy_score
from fuzzy_classifier import train_data

# variables
N_INDIV = 3
N_VARS = 4
N_CLASSES = 4

# conjunto difuso triangular
triangle_set = var.Var("triangle set")
triangle_set += "bajo", LMembership(-0.5, 0.5)
triangle_set += "medio", LMembership(0, 1)
triangle_set += "alto", LMembership(0.5, 1.5)

truthCache = {}
inferenceCache = {}


def toCacheString(rule, data_row):
    strRule = "".join(str(i) for i in rule)
    strRow = ""
    for x in range(len(data_row)):
        strRow += "%0.3f" % data_row.iloc[x]
    return strRule + strRow

# Generar reglas aleatorias
def generateRule():
    randBits = []
    randRule = randint(0, pow(2, 12)-1)
    rule = "{0:b}".format(randRule)

    # Modificar aqui para obtener + o - clases (target)
    randClass = randint(0,3)

    for i in range(12-len(rule)):
        randBits.append(0)
    for i in range(len(rule)):
        randBits.append(int(rule[i]))
    return randBits
    '''
    if (randClass == 0):
        randBits += [0,0,1]
    elif(randClass == 1):
        randBits += [0,1,0]
    else:
        randBits += [1,0,0]
    '''

def generateRules(n_indiv: int, n_vars: int, n_classes: int = 0):
    """
    genera una regla difusa aleatoria
    """
    randRule = randint(0, pow(2, n_vars * n_indiv) - 1)
    rule = "{0:b}".format(randRule)

    # Modificar aqui para obtener + o - clases (target)
    # randClass = randint(0,3)

    randBits = np.zeros(n_vars * n_indiv, int)
    for i in range(len(rule)):
        randBits[i] = int(rule[i])

    return randBits

def checkRules(indiv, X_data, y_data):
    # Obtener el puntaje de reglas buenas (joker regla) y reglas malas (sin clases)
    confVect = getConfVect(indiv.rules, X_data, y_data)
    goodRulesNb = 0
    badRulesNb = 0
    for classNb, conf in confVect:
        if classNb == -1:
            if conf == 0:
                badRulesNb += 1
            elif conf == -1:
                goodRulesNb += 1
            else:
                print("Esto es muy raro en classifier:revisarreglas")
    return goodRulesNb, badRulesNb

def getCompetitionStrength(rule, X, y, n_classes=N_CLASSES):
    """
    Devuelve la fuerza de competencia de la regla dada para cada clase
    """
    # Modificar aqui para obtener + o - clases (target)
    competitionStrength = np.zeros(n_classes, int)

    # Modificar aqui para obtener + o - clases (target)
    for classNumber in range(n_classes):
        classArray = train_data.getClassArray(classNumber, X, y)
        competitionStrength[classNumber] = sum(
            [getMuA(rule, row) for _, row in classArray.iterrows()]
        )

    return competitionStrength

# Esta funcion devuelve el numero de la clase y el porcentaje verdadero de esa clase
# basado en la funcion entrenamiento.
def getConf(rule, X, y):
    # Transformar [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1] en '001001010001'
    hashedRule = "".join(str(i) for i in rule)
    if hashedRule in truthCache:
        return truthCache[hashedRule]
    else:
        if hashedRule == "000000000000":
            maxIndex, maxValue = (-1, -1)
        else:
            # Revisar a traves de los datos de entrenamiento para obtener PesoCompeticion
            competitionStrength = getCompetitionStrength(rule, X, y)
            # Dividir por la suma para obtener porcentaje verdadero (entre 0 y 1)
            strSum = sum(competitionStrength)
            if strSum != 0:
                truthDegree = [i/strSum for i in competitionStrength]
                # Obtener la clase con el mejor valor, y su indice
                maxIndex, maxValue = max(enumerate(truthDegree),key=lambda x: x[1])
            else:
                maxIndex, maxValue = (-1, 0)  # Ninguna clase fue reconocida
        truthCache[hashedRule] = (maxIndex, maxValue)
        return (maxIndex, maxValue)

def getMuA(rule, data_row, fuzzy_set=triangle_set):
    """
    Devuelve el array de mutación para la regla dada y la fila de datos
    """
    # Revisar si la regla tiene una copia en el cache
    cacheString = toCacheString(rule, data_row)
    if cacheString in inferenceCache:
        return inferenceCache[cacheString]
    else:
        maxArray = []
        ruleCounter = 0
        for x in range(0, len(data_row)):
            datum = data_row.iloc[x]
            # revisar funciones de cada conjunto que esten activas
            if (
                rule[ruleCounter] != 0
                or rule[ruleCounter + 1] != 0
                or rule[ruleCounter + 2] != 0
            ):
                small = 0
                medium = 0
                large = 0
                if rule[ruleCounter] == 1:
                    small = fuzzy_set.sets["bajo"].membership(datum)
                if rule[ruleCounter + 1] == 1:
                    medium = fuzzy_set.sets["medio"].membership(datum)
                if rule[ruleCounter + 2] == 1:
                    large = fuzzy_set.sets["alto"].membership(datum)
                maxArray.append(max(small, medium, large))
            ruleCounter += 3
        if maxArray == []:
            muA = 0  #  * No estoy seguro de eso : significa que es la joker regla
        else:
            muA = min(maxArray)
        inferenceCache[cacheString] = muA
        return muA


def getPredictedConfVect(confVect, muAVect, n_classes=N_CLASSES):
    # Modificar aqui para obtener + o - clases (target)
    predictedConfVect = np.zeros(n_classes, int)
    cnt = np.ones(n_classes, int)

    for i in range(len(confVect)):
        ruleClass, ruleConf = confVect[i]
        if ruleClass != -1:
            predictedConfVect[ruleClass] += muAVect[i] * ruleConf
            cnt[ruleClass] += 1

    # Modificar aqui para obtener + o - clases (target)
    averagedPredictedConfVect = [predictedConfVect[i] / cnt[i] for i in range(3)]
    return averagedPredictedConfVect

def getMuAVect(rules, data_row):
    return [getMuA(rule, data_row) for rule in rules]

# Obtener la mejor clase y su porcentaje de confianza de cada regla
def getConfVect(rules, X, y):
    return [getConf(rule, X, y) for rule in rules]

def getPredictedClass(rules, X_data, y_data, data_row):
    predictedConfVect = getPredictedConfVect(getConfVect(rules, X_data, y_data), getMuAVect(rules, data_row))
    predictedClass, predictedConf = max(enumerate(predictedConfVect), key=lambda x: x[1])
    return predictedClass, predictedConf

def getPredictedClasses(indiv, X_data, y_data):
    predictedClassArray = []
    for _, data_row in X_data.iterrows():
        predictedClass, predictedConf = getPredictedClass(indiv.rules, X_data, y_data, data_row)
        predictedClassArray.append(predictedClass)
    return predictedClassArray


def getAccuracy(indiv, X, y):
    predictedClassArray = getPredictedClasses(indiv, X, y)
    score = accuracy_score(y, predictedClassArray)
    return score

def calcComplexity(indiv, dontCare=True):
    complexity = 0
    for rule in indiv.rules:
        if dontCare:
            for i in range(0, 3):
                if not (rule[i] == rule[i+1] == rule[i+2] == 0):
                    complexity += 1
                    i += 3
        else:
            complexity += sum(rule)
    return complexity