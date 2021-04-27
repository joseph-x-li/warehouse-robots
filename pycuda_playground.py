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


kernel = f"""
__global__ void step(float *rewards, int *actions, int *poss, int *goals, int *field){{
    const int tidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int movlookup_r[4] = {{ -1, 0, 1,  0 }};
    const int movlookup_c[4] = {{  0, 1, 0, -1 }};
    int oldposctr = 0;
    int oldpos_r[1000];
    int oldpos_c[1000];
    float reward = 0.0;
    for(int i = tidx; i < {n_agents}; i += nthreads){{
        int action = actions[i];
        int currpos[2];
        int goal[2];
        currpos[0] = poss[i * 2];
        currpos[1] = poss[(i * 2) + 1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];
        int nextpos[2];
        if(action == 0){{// a no-op
            nextpos[0] = currpos[0];
            nextpos[1] = currpos[1];
        }} else {{ 
            nextpos[0] = currpos[0] + movlookup_r[action - 1];
            nextpos[1] = currpos[1] + movlookup_c[action - 1];

            if(nextpos[0] < 0 || nextpos[1] < 0 || nextpos[0] >= {rows} || nextpos[1] >= {cols}){{
                reward += {WALL_COLLISION_REWARD};
                nextpos[0] = currpos[0];
                nextpos[1] = currpos[1];
            }} else if(atomicCAS(&field[nextpos[0] * {cols} + nextpos[1]], 0, 1) == 1) {{
                reward += {ROBOT_COLLISION_REWARD};
                nextpos[0] = currpos[0];
                nextpos[1] = currpos[1];
            }}
        }}

        reward += abs(currpos[0] - goal[0]) - abs(nextpos[0] - goal[0]);
        reward += abs(currpos[1] - goal[1]) - abs(nextpos[1] - goal[1]);
        if(nextpos[0] == goal[0] && nextpos[1] == goal[1]){{
            reward += {GOAL_REWARD};
        }}
        if(nextpos[0] != currpos[0] || nextpos[1] != currpos[1]){{
            poss[i * 2] = nextpos[0];
            poss[i * 2 + 1] = nextpos[1];
            oldpos_r[oldposctr] = currpos[0];
            oldpos_c[oldposctr] = currpos[1];
            oldposctr += 1;
        }}
        rewards[i] = reward;

    }}
    __syncthreads();

    for(int i = 0; i < oldposctr; i++){{
        int r = oldpos_r[i];
        int c = oldpos_c[i];
        field[r * {cols} + c] = 0;
    }}
}}"""

print(kernel)

mod = SourceModule(kernel)
step = mod.get_function("step")