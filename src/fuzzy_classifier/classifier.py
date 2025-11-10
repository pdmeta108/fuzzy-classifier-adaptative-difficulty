import data
from random import randint
import numpy as np
from sklearn.metrics import accuracy_score

#### VARIABLES GA ####
mutProb = .1
tournamentSize = 4
elitism = True
######################

WEIGHT_ERROR = 0.2

alpha_cut = 0

nbRules = 8

truthCache = {}
inferenceCache = {}


# Clase para computar funciones Triangulo
class Triangle:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.alpha_cut = alpha_cut

    def valAt(self,x):
        alpha = 2/(self.max-self.min)
        f = lambda x: alpha * (x - self.min)
        g = lambda x: alpha * (-x + self.max)
        mid = (self.min + self.max)/2
        if x < self.min or x > self.max:
            return 0.0
        elif x < mid:
            return f(x)
        else:
            return g(x)

    def at(self, x):
        val = self.valAt(x)
        return val if val >= alpha_cut else 0


# Clase para funcion Trapezoide
class Trapezoid:
    def __init__(self, min, max_left, max_right, max):
        self.min = min
        self.max_left = max_left
        self.max_right = max_right
        self.max = max
        self.alpha_cut = alpha_cut

    def valAt(self, x):
        alpha_left = 1 / (self.max_left - self.min)
        alpha_right = 1 / (self.max - self.max_right)
        f = lambda x: alpha_left * (x - self.min)
        g = lambda x: alpha_right * (-x + self.max)
        if x < self.min or x >= self.max:
            return 0.0
        elif self.max_left <= x <= self.max_right:
            return 1.0
        elif self.min <= x < self.max_left:
            return f(x)
        elif self.max_right < x < self.max:
            return g(x)

    def at(self, x):
        val = self.valAt(x)
        return val if val >= alpha_cut else 0

# Por el momento todos los parametros comparten las mismas funciones miembro


#### FUNCIONES MIEMBRO ####
# X se encuentra solo en [0;1]
smallTriangle = Triangle(-0.5, 0.5)
medTriangle = Triangle(0, 1)
largeTriangle = Triangle(0.5, 1.5)
##############################


class Indiv:
    def __init__(self):
        self.rules = []
        for i in range(nbRules):
            self.rules.append(generateRule())

    def __str__(self):
        s = ""
        for i in range(nbRules):
            s += "Rule "+str(i)+": " + self.rules[i] + "\n"
        return s


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


def getCompetitionStrength(rule):
    # Modificar aqui para obtener + o - clases (target)
    competitionStrength = [0, 0, 0, 0]

    # Modificar aqui para obtener + o - clases (target)
    for classNumber in range(0,4):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] = sum([getMuA(rule, row) for _, row in classArray.iterrows()])

    return competitionStrength


# Esta funcion devuelve el numero de la clase y el porcentaje verdadero de esa clase basado en la funcion entrenamiento.
def getConf(rule):
    # Transformar [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1] en '001001010001'
    hashedRule = "".join(str(i) for i in rule)
    if hashedRule in truthCache:
        return truthCache[hashedRule]
    else:
        if hashedRule == "000000000000":
            maxIndex, maxValue = (-1, -1)
        else:
            # Revisar a traves de los datos de entrenamiento para obtener PesoCompeticion
            competitionStrength = getCompetitionStrength(rule)
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


# Obtener la mejor clase y su porcentaje de confianza de cada regla
def getConfVect(rules):
    return [getConf(rule) for rule in rules]


def toCacheString(rule,data_row):
    strRule = "".join(str(i) for i in rule)
    strRow = ""
    for x in range(len(data_row)):
        strRow += "%0.3f" % data_row[x]
    return strRule+strRow


def getMuA(rule,data_row):
    cacheString = toCacheString(rule,data_row)
    if cacheString in inferenceCache:
        return inferenceCache[cacheString]
    else:
        maxArray = []
        ruleCounter = 0
        for x in range(0, len(data_row)):
            datum = data_row[x]
            if rule[ruleCounter:ruleCounter+3] != [0, 0, 0]:
                small = 0
                medium = 0
                large = 0
                if rule[ruleCounter] == 1:
                    small = smallTriangle.at(datum)
                if rule[ruleCounter+1] == 1:
                    medium = medTriangle.at(datum)
                if rule[ruleCounter+2] == 1:
                    large = largeTriangle.at(datum)
                maxArray.append(max(small, medium, large))
            ruleCounter += 3
        if maxArray == []:
            muA = 0  # No estoy seguro de eso : significa que es la joker regla
        else:
            muA = min(maxArray)
        inferenceCache[cacheString] = muA
        return muA


def getMuAVect(rules, data_row):
    return [getMuA(rule, data_row) for rule in rules]


def getPredictedConfVect(confVect, muAVect):
    # Modificar aqui para obtener + o - clases (target)
    predictedConfVect = [0, 0, 0, 0]
    cnt = [1, 1, 1, 1]

    for i in range(len(confVect)):
        ruleClass, ruleConf = confVect[i]
        if ruleClass != -1:
            predictedConfVect[ruleClass] += muAVect[i] * ruleConf
            cnt[ruleClass] += 1

    # Modificar aqui para obtener + o - clases (target)
    averagedPredictedConfVect = [predictedConfVect[i]/cnt[i] for i in range(3)]
    return averagedPredictedConfVect

def getPredictedClass(rules, data_row):
    predictedConfVect = getPredictedConfVect(getConfVect(rules), getMuAVect(rules, data_row))
    predictedClass, predictedConf = max(enumerate(predictedConfVect), key=lambda x: x[1])
    return predictedClass, predictedConf


def getPredictedClasses(indiv, data):
    predictedClassArray = []
    for _, data_row in data.iterrows():
        predictedClass, predictedConf = getPredictedClass(indiv.rules, data_row)
        predictedClassArray.append(predictedClass)
    return predictedClassArray


def getAccuracy(indiv):
    predictedClassArray = getPredictedClasses(indiv, data.X_test)
    score = accuracy_score(data.y_test, predictedClassArray)
    return score

def checkRules(indiv):
    # Obtener el puntaje de reglas buenas (joker regla) y reglas malas (sin clases)
    confVect = getConfVect(indiv.rules)
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

# Computar µ_a dado un u_i y una regla
def getMuAPast(muArray,rule):

    # Multiplicar muArray con la regla para "eliminar" los set difusos sin usar

    maxArray = []

    for i in [0, 3, 6, 9]:
        if rule[i:i+3] != [0, 0, 0]:  # No importa acerca de este parametro
            maxArray.append(max([muArray[j] * rule[j] for j in range(i, i+3)]))

    # muA = min( [ max(muValues[i:i+3]) for i in [0,3,6,9] ] )
    if maxArray == []:  # Regla incorrecta?
        return -1
    muA = min(maxArray)
    return muA


# Capaz se prefiera preprocesar muArray para todos los elementos
# Para no computar cada vez

# Devuelve la clase predicha por la regla
def getClassFromRule(rule):
    # Se computa qué tan bien se da esa clase para esta regla
    competitionStrength = [0, 0, 0]
    # Modificar aqui para obtener + o - clases (target)
    for classNumber in range(3):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] += np.dot(classArray,rule)
    maxIndex = getMaxIndex(competitionStrength)
    print("La clase para esta regla es la clase " + str(maxIndex) + \
          " con un puntaje: " + str(competitionStrength[maxIndex]))
    return maxIndex, competitionStrength


# o solo solo usar max(enumerate(a),key=lambda x: x[1])[0]
def getMaxIndex(l):
    if len(l) == 1:
        return 0
    else:
        max = l[0]
        index = 0
        for i in range(1,len(l)):
            if l[i] > max:
                max = l[i]
                index = i
        return index

