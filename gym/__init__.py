import gym.warehouse_complex
import gym.warehouse_simple

lookup = {
    "warehouse_simple": warehouse_simple.Gym,
    "warehouse_complex": warehouse_complex.Gym,
}

def make(env):
    return lookup[env]()