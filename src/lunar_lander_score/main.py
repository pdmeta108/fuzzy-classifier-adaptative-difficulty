import sys
import numpy as np
from random import randint, random, uniform

# setting path
sys.path.append('../')

from fuzzy_classifier.train_data import getTrainedLanderData
from fuzzy_classifier import generate_rule_classifier

#### Numero de reglas base ####
nbRules = 8

# variables de regla
N_INDIV = 3
N_VARS = 4
N_CLASSES = 4

#### VARIABLES GA ####
mutProb = .1
tournamentSize = 3
elitism = True
popSize = 15
GENERATION_NUMBER = 20
# ------------------- #
# Peso de los parametros de funcion para calcular fitness
# ALPHA = 0.6
# BETA = 0.4

# Crear clases Individuo y Poblacion

class Indiv:
    def __init__(self, init=True):
        self.rules = []
        if init:
            for i in range(nbRules):
                self.rules.append(generate_rule_classifier.generateRule())

    def __str__(self, data_X=None, data_y=None):
        s = ""
        for i in range(nbRules):
            s += "Regla " + str(i) + ": " + str(self.rules[i]) + "\t"
            s += str(generate_rule_classifier.getConf(self.rules[i]), data_X, data_y) + "\n"
        return s

    def getFitness(self, data_X, data_y):
        acc = generate_rule_classifier.getAccuracy(self, data_X, data_y)
        goodRulesNb, badRulesNb = generate_rule_classifier.checkRules(self)
        complexity = generate_rule_classifier.calcComplexity(self)

        w1 = 0.6
        w2 = 0.4

        score = w1 * (1.0 - acc) + w2 * (float(complexity) / float(len(self.rules) * 4))
        # this maximizes the accuracy and minimizes "complexity"
        score = -1 * score

        return score

        # 1 - nbofaccuratelyclassified / number of cases find

        # inferences = generate_rule_classifier.infer(self.rules)
        # inferences = generate_rule_classifier.simple_infer(self.rules)
        # return generate_rule_classifier.computeFitness(inferences)


class Population:
    """ x = Indiv()
    x.getfit()"""

    def __init__(self, init, size=popSize):
        if init:
            self.listpop = [Indiv() for _ in range(size)]
        else:
            self.listpop = []

    def getFittest(self, data_X, data_y):
        nb_max = self.listpop[0].getFitness(data_X, data_y)
        index = 0

        for i in range(1, len(self.listpop)):
            nextFitness = self.listpop[i].getFitness(data_X, data_y)
            if nextFitness > nb_max:
                nb_max = nextFitness
                index = i
        return self.listpop[index]

# FUNCIONES PARA PROCESO DE ALGORITMO GENETICO
def tournament(pop, data_X, data_y):
    tourList = Population(False, tournamentSize)

    for j in range(tournamentSize):
        indexT = randint(0, popSize - 1)

        pop.listpop[indexT]
        tourList.listpop.append(pop.listpop[indexT])

    return tourList.getFittest(data_X, data_y)


def crossOver(Indiv1, Indiv2):
    newIndiv = Indiv(False)

    for i in range(nbRules):
        rule1 = Indiv1.rules[i]

        rule2 = Indiv2.rules[i]

        newIndiv.rules.append(crossoverRules(rule1, rule2))

    return newIndiv


def crossoverRules(rule1, rule2):
    newRule = []

    for i in range(len(rule1)):
        prob = random()
        if prob < 0.5:
            newRule.append(rule1[i])

        else:
            newRule.append(rule2[i])

    return newRule


def mutation(indiv):
    for i in range(nbRules):

        for j in range(nbRules):

            prob = random()

            if prob < mutProb:
                indiv.rules[i][j] = 1 - indiv.rules[i][j]


# Obtener set de reglas y salidas del resultado
def getRulesfromFittest(thisFittest, data_X, data_y):
    reglas = []
    salidas = []
    for i in range(len(thisFittest.rules)):
        reglas.append(thisFittest.rules[i])
        salidas.append(generate_rule_classifier.getConf(thisFittest.rules[i], data_X, data_y))
    return reglas, salidas


# Insertar datos de la sub clase correpondiente a la clase de la regla
def getRulesSubClass(rule, i, ruleclass, subclass):
    rule_text = ""
    if not (rule[i] == 0) and not ((rule[i + 1] == 0) or (rule[i + 2] == 0)):
        rule_text += ruleclass + " = " + subclass + " or "
    elif not (rule[i] == 0) and (rule[i + 1] == 0) and (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass + " and "
    if not (rule[i + 1] == 0) and not (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass + " or "
    elif not (rule[i + 1] == 0) and (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass + " and "
    if not (rule[i + 2] == 0):
        rule_text += ruleclass + " = " + subclass + " and "
    return rule_text


# Transformar reglas binarias en reglas literales
def binRulestoClassRules (rules, outputs, entrada, salida, subclase):
    soltext = []
    count = 0
    for rule in rules:
        rule_text = ""
        textosalida = ""
        for i in range(0, len(rule), 3):
            if i < 2:
                rule_text += getRulesSubClass(rule, i, entrada[0], subclase[0])
            elif 2 <= i < 5:
                rule_text += getRulesSubClass(rule, i, entrada[1], subclase[1])
            elif 5 <= i < 8:
                rule_text += getRulesSubClass(rule, i, entrada[2], subclase[2])
            elif 8 <= i < 11:
                rule_text += getRulesSubClass(rule, i, entrada[3], subclase[3])
        rule_text_2 = rule_text.rstrip('and ')
        # Output process
        tupla = outputs[count]
        if tupla[0] == 0:
            textosalida += salida[0]
        elif tupla[0] == 1:
            textosalida += salida[1]
        elif tupla[0] == 2:
            textosalida += salida[2]
        else:
            textosalida += "Not Valid"
        rule_text_2 += " entonces " + textosalida
        soltext.append(rule_text_2)
        count += 1
    return soltext


# Obtener valor de parametros de subclase
def getSubClassValue(parametro_box, subclase):
    if parametro_box == subclase[0]:
        return round(uniform(0, 0.25), 3)
    elif parametro_box == subclase[1]:
        return round(uniform(0.25, 0.75), 3)
    elif parametro_box == subclase[2]:
        return round(uniform(0.75, 1), 3)
    else:
        return 0

if __name__ == "__main__":
    # print("variable triangulo", triangle_set)

    # Entrenamiento de datos
    X_train, X_test, y_train, y_test = getTrainedLanderData()
    # Crear clase poblacion para comenzar a generar reglas

    pop = Population(True)
    # Se realiza en cada generacion el proceso de torneo y mutacion

    fit_rules = []

    for i in range(GENERATION_NUMBER):

        newpop = Population(False)
        print("Generacion numero : {}".format(str(i)))

        for j in range(popSize):
            # Proceso de torneo
            parent1 = tournament(pop, X_test, y_test)
            parent2 = tournament(pop, X_test, y_test)

            child = crossOver(parent1, parent2)
            newpop.listpop.append(child)

        for j in range(popSize):
            # Proceso de mutacion
            mutation(newpop.listpop[j])

        pop = newpop
        # Mostrar el mejor set de reglas en la poblacion
        thisFittest = pop.getFittest(X_test, y_test)
        print("Mejor precision ajustada : {}".format(str(generate_rule_classifier.getAccuracy(thisFittest, X_test, y_test))))
        print(thisFittest)
        fit_rules.append(generate_rule_classifier.getAccuracy(thisFittest, X_test, y_test))

    # Mostrar la complejidad del set de reglas
    print("Calculo de complejidad final: {}".format(generate_rule_classifier.calcComplexity(thisFittest)))
    # Obtener set de reglas con salida y guardar en Main()
    reglas, salidas = getRulesfromFittest(thisFittest, X_test, y_test)
    # rule_class = generateRules(N_INDIV, N_VARS)
    # print(rule_class, type(rule_class))
    # print(X_test.loc[0], type(X_test.loc[0]))
    # print(triangle_set.sets["medio"].membership(300))
    # array_mt = getMutationArray(rule_class, X_test.loc[0], triangle_set)
    # array_mt = getCompetitionStrength(rule_class, N_CLASSES, X_train, y_train)
    # print("competition strength", array_mt)
