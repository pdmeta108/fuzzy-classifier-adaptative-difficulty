from inferfuzzy import var
from inferfuzzy.membership import Membership
from inferfuzzy.memberships import (
    LMembership,
)
from inferfuzzy.systems import MamdaniSystem
from inferfuzzy.defuzzifications import centroid_defuzzification

class SingletonMembership(Membership):
    """Clase de membresia singleton, que devuelve 1 si el valor es igual a 'a', y 0 en caso contrario

    Parameters
    ----------
    Membership : _type_
        Superclase Membership, que se encarga de definir la interfaz de las funciones de membresia
    """
    def __init__(self, a):
        def func(x):
            return 1.0 if x == a else 0.0

        super(SingletonMembership, self).__init__(func, [a])

def get_lunar_lander_inference_system():
    # conjunto difuso triangular
    time_set = var.Var("time")
    time_set += "bajo", LMembership(-100, 100)
    time_set += "medio", LMembership(0, 200)
    time_set += "alto", LMembership(100, 300)
    print("variable tiempo", time_set)

    reward_set = var.Var("reward")
    reward_set += "bajo", LMembership(-60, 180)
    reward_set += "medio", LMembership(0, 240)
    reward_set += "alto", LMembership(180, 360)
    print("variable puntuacion", reward_set)

    win_set = var.Var("win")
    win_set += "bajo", LMembership(-0.5, 0.5)
    win_set += "medio", LMembership(0, 1)
    win_set += "alto", LMembership(0.5, 1.5)
    print("variable exito", win_set)

    gravity_set = var.Var("gravity")
    gravity_set += "bajar", SingletonMembership(0)
    gravity_set += "subir", SingletonMembership(100)

    # Sistema de Inferencia difusa
    mamdani = MamdaniSystem(
        defuzz_func=centroid_defuzzification,
    )

    # Reglas difusas
    mamdani += (
        time_set.into("bajo")
        & reward_set.into("alto")
        & win_set.into("alto")
    ), gravity_set.into("subir")

    mamdani += (
        time_set.into("alto")
        & win_set.into("bajo")
    ), gravity_set.into("bajar")
    return mamdani