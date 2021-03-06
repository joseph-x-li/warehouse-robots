import pycuda.autoinit
from pycuda.compiler import SourceModule


def stepkernel(
    rows,
    cols,
    nagents,
    WALL_COLLISION_REWARD,
    ROBOT_COLLISION_REWARD,
    GOAL_REWARD,
):
    kernel = f""" \
__global__ void step(float *rewards, int *actions, int *poss, int *goals, int *field){{
    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;
    const int nblocks = gridDim.x;

    const int movlookup_r[4] = {{ -1, 0, 1,  0 }};
    const int movlookup_c[4] = {{  0, 1, 0, -1 }};
    int oldposctr = 0;
    int oldpos_r[1000];
    int oldpos_c[1000];
    
    int start = blockidx * nthreads + threadidx;
    int step = nblocks * nthreads;

    for(int i = start; i < {nagents}; i += step){{
        float reward = 0.0;
        int action = actions[i];
        int currpos[2];
        int initpos[2];
        int goal[2];
        currpos[0] = poss[i * 2];
        currpos[1] = poss[(i * 2) + 1];
        initpos[0] = currpos[0];
        initpos[1] = currpos[1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];
        int nextpos[2];
        if(action == 0){{// a no-op
            nextpos[0] = currpos[0];
            nextpos[1] = currpos[1];
        }} else {{  // Single Step n times
            action -= 1;
            while(action >= 0){{
                nextpos[0] = currpos[0] + movlookup_r[action % 4];
                nextpos[1] = currpos[1] + movlookup_c[action % 4];

                if(nextpos[0] < 0 || nextpos[1] < 0 || nextpos[0] >= {rows} || nextpos[1] >= {cols}){{
                    reward += {WALL_COLLISION_REWARD};
                    nextpos[0] = currpos[0];
                    nextpos[1] = currpos[1];
                }} else if(atomicCAS(&field[nextpos[0] * {cols} + nextpos[1]], 0, 1) == 1) {{
                    reward += {ROBOT_COLLISION_REWARD};
                    nextpos[0] = currpos[0];
                    nextpos[1] = currpos[1];
                }}
                if(nextpos[0] != currpos[0] || nextpos[1] != currpos[1]){{
                    oldpos_r[oldposctr] = currpos[0];
                    oldpos_c[oldposctr] = currpos[1];
                    oldposctr += 1;
                }}
                // update currpos
                currpos[0] = nextpos[0];
                currpos[1] = nextpos[1];
                action -= 4;
            }}
            poss[i * 2] = nextpos[0];
            poss[i * 2 + 1] = nextpos[1];
        }}

        // reward only depends on initial position and end position
        reward += abs(initpos[0] - goal[0]) - abs(nextpos[0] - goal[0]);
        reward += abs(initpos[1] - goal[1]) - abs(nextpos[1] - goal[1]);
        if(nextpos[0] == goal[0] && nextpos[1] == goal[1]){{
            reward += {GOAL_REWARD};
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
    return SourceModule(kernel).get_function("step")


def tensorkernel(rows, cols, view_size, nagents):
    kernel = f""" \
__global__ void tensor(float *states, int *poss, int *goals, int *field){{
    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;
    const int nblocks = gridDim.x;
    const int view_range = {view_size // 2};
    const int displacement = {view_size ** 2};
    const int statesize = {view_size ** 2 + 4};

    float tmpstate[statesize];
    int pos[2];
    int goal[2];

    int start = blockidx * nthreads + threadidx;
    int step = nblocks * nthreads;

    for(int i = start; i < {nagents}; i += step){{
        pos[0] = poss[i * 2];
        pos[1] = poss[(i * 2) + 1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];

        tmpstate[displacement] = (2 * ((float)pos[0]) / {rows}) - 1;
        tmpstate[displacement + 1] = (2 * ((float)pos[1]) / {cols}) - 1;
        tmpstate[displacement + 2] = (2 * ((float)goal[0]) / {rows}) - 1;
        tmpstate[displacement + 3] = (2 * ((float)goal[1]) / {cols}) - 1;
        
        // states[i * statesize + displacement] = (2 * ((float)pos[0]) / {rows}) - 1;
        // states[i * statesize + displacement + 1] = (2 * ((float)pos[1]) / {cols}) - 1;
        // states[i * statesize + displacement + 2] = (2 * ((float)goal[0]) / {rows}) - 1;
        // states[i * statesize + displacement + 3] = (2 * ((float)goal[1]) / {cols}) - 1;
        
        // takes chunks of updated fields and updates the nagents receptive fields
        // don't want to break field into sections because obstacles in field aren't represented
        // accurately in pos
        for(int r = 0; r < {view_size}; r++){{
            for(int c = 0; c < {view_size}; c++){{
                int fieldr = pos[0] + r - view_range;
                int fieldc = pos[1] + c - view_range;
                float fillval = -1;
                if(fieldr >= 0 && fieldc >= 0 && fieldr < {rows} && fieldc < {cols})
                    fillval = (float)field[fieldr * {cols} + fieldc];
                tmpstate[r * {view_size} + c] = fillval;
                // int stateidx = statesize * i + r * {view_size} + c;
                // states[stateidx] = fillval;
            }}
        }}

        __syncthreads();

        for(int j = 0; j < statesize; j++) {{
            states[statesize * i + j] = tmpstate[j];
        }}
    }}
}}"""
    return SourceModule(kernel).get_function("tensor")


# def megastepkernel(
#     envs,
#     rows,
#     cols,
#     nagents,
#     WALL_COLLISION_REWARD,
#     ROBOT_COLLISION_REWARD,
#     GOAL_REWARD,
#     extended=False,
# ):
#     kernel = f""" \
# __global__ void step(float *rewards, int *actions, int *poss, int *goals, int *field){{
#     const int threadidx = threadIdx.x;
#     const int nthreads = blockDim.x;
#     const int movlookup_r[4] = {{ -1, 0, 1,  0 }};
#     const int movlookup_c[4] = {{  0, 1, 0, -1 }};
#     int oldposctr = 0;
#     int oldpos_e[10000];
#     int oldpos_r[10000];
#     int oldpos_c[10000];
#     for(int i = threadidx; i < {nagents}; i += nthreads){{
#         float reward = 0.0;
#         int action = actions[i];
#         int currpos[2];
#         int initpos[2];
#         int goal[2];
#         currpos[0] = poss[i * 2];
#         currpos[1] = poss[(i * 2) + 1];
#         initpos[0] = currpos[0];
#         initpos[1] = currpos[1];
#         goal[0] = goals[i * 2];
#         goal[1] = goals[(i * 2) + 1];
#         int nextpos[2];
#         if(action == 0){{// a no-op
#             nextpos[0] = currpos[0];
#             nextpos[1] = currpos[1];
#         }} else {{  // Single Step n times
#             action -= 1;
#             while(action >= 0){{
#                 nextpos[0] = currpos[0] + movlookup_r[action % 4];
#                 nextpos[1] = currpos[1] + movlookup_c[action % 4];

#                 if(nextpos[0] < 0 || nextpos[1] < 0 || nextpos[0] >= {rows} || nextpos[1] >= {cols}){{
#                     reward += {WALL_COLLISION_REWARD};
#                     nextpos[0] = currpos[0];
#                     nextpos[1] = currpos[1];
#                 }} else if(atomicCAS(&field[nextpos[0] * {cols} + nextpos[1]], 0, 1) == 1) {{
#                     reward += {ROBOT_COLLISION_REWARD};
#                     nextpos[0] = currpos[0];
#                     nextpos[1] = currpos[1];
#                 }}
#                 if(nextpos[0] != currpos[0] || nextpos[1] != currpos[1]){{
#                     oldpos_r[oldposctr] = currpos[0];
#                     oldpos_c[oldposctr] = currpos[1];
#                     oldposctr += 1;
#                 }}
#                 // update currpos
#                 currpos[0] = nextpos[0];
#                 currpos[1] = nextpos[1];
#                 action -= 4;
#             }}
#             poss[i * 2] = nextpos[0];
#             poss[i * 2 + 1] = nextpos[1];
#         }}

#         // reward only depends on initial position and end position
#         reward += abs(initpos[0] - goal[0]) - abs(nextpos[0] - goal[0]);
#         reward += abs(initpos[1] - goal[1]) - abs(nextpos[1] - goal[1]);
#         if(nextpos[0] == goal[0] && nextpos[1] == goal[1]){{
#             reward += {GOAL_REWARD};
#         }}
#         rewards[i] = reward;

#     }}
#     __syncthreads();

#     for(int i = 0; i < oldposctr; i++){{
#         int r = oldpos_r[i];
#         int c = oldpos_c[i];
#         field[r * {cols} + c] = 0;
#     }}
# }}"""
#     return SourceModule(kernel).get_function("step")
