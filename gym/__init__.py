import gym.warehouse_complex
import gym.warehouse_simple
import gym.warehouse_simple_multi
import gym.warehouse_simple_multi_gpu
import gym.warehouse_simple_mega_gpu

lookup = {
    "warehouse_simple": warehouse_simple.Gym,
    "warehouse_simple_multi": warehouse_simple_multi.Gym,
    "warehouse_simple_multi_gpu": warehouse_simple_multi_gpu.Gym,
    "warehouse_complex": warehouse_complex.Gym,
    "warehouse_simple_mega_gpu": warehouse_simple_mega_gpu.Gym,
}


def make(env):
    try:
        return lookup[env]()
    except KeyError as e:
        print(f"Unrecgonized gym {env}. Available gyms: {lookup.keys()}")
        return None
