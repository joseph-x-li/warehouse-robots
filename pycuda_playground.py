import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData

dd = DeviceData()
print(dd.shared_memory)
print(dd.max_threads)

rows = 10
cols = 10 
n_agents = 10
WALL_COLLISION_REWARD = -1.1 
ROBOT_COLLISION_REWARD = -3
GOAL_REWARD = 2
view_size = 11





kernel = f"""__global__ void tensor(int *states, int *poss, int *goals, int *field){{
    const int tidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int view_range = {view_size // 2};
    const int displacement = {view_size ** 2};
    const int statesize = {view_size ** 2 + 4};
    for(int i = tidx; i < {n_agents}; i += nthreads){{
        int pos[2];
        int goal[2];
        pos[0] = poss[i * 2];
        pos[1] = poss[(i * 2) + 1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];
        states[i * statesize + displacement] = (2 * pos[0] / {rows}) + 1;
        states[i * statesize + displacement + 1] = (2 * pos[1] / {cols}) + 1;
        states[i * statesize + displacement + 2] = (2 * goal[0] / {rows}) + 1;
        states[i * statesize + displacement + 3] = (2 * goal[1] / {cols}) + 1;
        for(int r = 0; r < {view_size}; r++){{
            for(int c = 0; c < {view_size}; c++){{
                int fieldr = pos[0] + r - view_range;
                int fieldc = pos[1] + c - view_range;
                int oob = 0;
                if(fieldr < 0 || {rows} <= fieldr)
                    oob = 1;
                if(fieldc < 0 || {cols} <= fieldc)
                    oob = 1;
                int fillval = -1;
                if(oob == 0)
                    fillval = field[fieldr * {rows} + fieldc];
                int stateidx = statesize * i + r * {view_size} + c;
                states[stateidx] = fillval;
            }}
        }}
    }}
}}
"""
print(kernel)

mod = SourceModule(kernel)
step = mod.get_function("tensor")