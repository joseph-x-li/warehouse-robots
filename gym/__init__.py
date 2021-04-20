import gym.warehouse_complex
import gym.warehouse_simple
import gym.warehouse_simple_multi

lookup = {
    "warehouse_simple": warehouse_simple.Gym,
    "warehouse_simple_multi": warehouse_simple_multi.Gym,
    "warehouse_complex": warehouse_complex.Gym,
}

def make(env):
    try:
        return lookup[env]()
    except KeyError as e:
        print(f"Unrecgonized gym {env}. Available gyms: {lookup.keys()}")
        return None