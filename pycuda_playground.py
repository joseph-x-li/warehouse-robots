import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData
from math import ceil

dd = DeviceData()
print(dd.shared_memory)
print(dd.max_threads)
print(dir(dd))

rows = 10
cols = 10 
nagents = 10
WALL_COLLISION_REWARD = -1.1 
ROBOT_COLLISION_REWARD = -3
GOAL_REWARD = 2
view_size = 11
binwidth = 11
binr = ceil(rows / binwidth)
binc = ceil(cols / binwidth)
nbins = binr * binc


kernel = f""" \
__global__ void tensor(float *states, int *poss, int *goals, int *field){{
    __shared__ int bincounters[{nbins}];
    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;
    const int nblocks = gridDim.x;

    const int view_range = {view_size // 2};
    const int displacement = {view_size ** 2};
    const int statesize = {view_size ** 2 + 4};

    int myposs[100][2];
    int mygoals[100][2];

    int start = blockidx * nthreads + threadidx;
    int step = nblocks * nthreads;

    for(int i = start; i < {nbins}; i += step){{
        bincounters[i] = 0;
    }}
    __syncthreads();

    for(int i = start; i < {nagents}; i += step){{
        pos[0] = poss[i * 2];
        pos[1] = poss[(i * 2) + 1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];
        states[i * statesize + displacement] = (2 * ((float)pos[0]) / {rows}) - 1;
        states[i * statesize + displacement + 1] = (2 * ((float)pos[1]) / {cols}) - 1;
        states[i * statesize + displacement + 2] = (2 * ((float)goal[0]) / {rows}) - 1;
        states[i * statesize + displacement + 3] = (2 * ((float)goal[1]) / {cols}) - 1;
        int mybin = (pos[0] / {binwidth}) *  + ;
        atomicAdd(&(bincounters[mybin]), 1);
    }}
}}"""
print(kernel)

mod = SourceModule(kernel, no_extern_c=True)
step = mod.get_function("tensor")