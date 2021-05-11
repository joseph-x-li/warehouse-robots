import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData

dd = DeviceData()
print(dd.shared_memory)
print(dd.max_threads)
print(dir(dd))

rows = 1000
cols = 1000
nagents = 10000
WALL_COLLISION_REWARD = -1.1 
ROBOT_COLLISION_REWARD = -3
GOAL_REWARD = 2
view_size = 11
binwidth = 100
from math import ceil
binr = ceil(rows / binwidth)
binc = ceil(cols / binwidth)
nbins = binr * binc

def findNextPowerOf2(n):
    n = n - 1
    while n & n - 1: n = n & n - 1 
    return n << 1
nbins_rounded = findNextPowerOf2(nbins)


kernel = f""" \
__device__ __inline__ void exclusiveScan(int length, int *array) {{
    const int threadidx = threadIdx.x;
    // upsweep
    for (int step = 1; step < length; step *= 2) {{
        // based on step, compute the position of the two active elements
        int idxA = (step - 1) + (2 * step * threadidx);
        int idxB = idxA + step;
        if(idxB < length) {{
            array[idxB] += array[idxA];
        }}
        __syncthreads();
    }}
    if(threadidx == 0){{ array[length - 1] = 0; }}
    __syncthreads();
    // downsweep
    for (int step = length/2; step >= 1; step /= 2) {{
        // based on step, compute the position of the two active elements
        int idxA = (step - 1) + (2 * step * threadidx);
        int idxB = idxA + step;
        if(idxB < length) {{
            int hold = array[idxB];
            array[idxB] += array[idxA];
            array[idxA] = hold;
        }}
        __syncthreads();
    }}
}}

__global__ void tensor(float *all_states, int *all_poss, int *all_goals){{
    // MAX SHARED MEMORY: 49152 BYTES
    __shared__ int bincounters[{nbins_rounded}];  // index i is number of agents in bin i
    __shared__ int binpfx[{nbins_rounded}];       // prefix sums of the above (starts from 0)
    __shared__ unsigned short bins[{nagents}][2]; // where the agent locations will actually be stored

    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;
    
    const int view_range = {view_size // 2};
    const int displacement = {view_size ** 2};
    const int statesize = {view_size ** 2 + 4};

    float *states = &(all_states[blockidx * {nagents} * statesize]);
    int *poss = &(all_poss[blockidx * {nagents * 2}]);
    int *goals = &(all_goals[blockidx * {nagents * 2}]);

    int start = threadidx;
    int step = nthreads;

    for(int i = start; i < {nbins_rounded}; i += step){{
        bincounters[i] = 0;
        binpfx[i] = 0;
    }}
    __syncthreads();

    // For positions this thread is responsible for, 
    // determine its bin index
    // increment that bin counter
    // remember its place in that bin and its bin assignment
    int localctr = 0;   // counts loop iterations to index into myposs
    int myposs[100][5]; // positions I am responsible for [0, 1] = r, c; [2, 3] = bin_idx, bin_pos; [4] = index in poss
    int goal[2];        // goal value holder
    for(int i = start; i < {nagents}; i += step){{
        myposs[localctr][0] = poss[i * 2];
        myposs[localctr][1] = poss[(i * 2) + 1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];
        states[i * statesize + displacement] = (2 * ((float)myposs[localctr][0]) / {rows}) - 1;
        states[i * statesize + displacement + 1] = (2 * ((float)myposs[localctr][1]) / {cols}) - 1;
        states[i * statesize + displacement + 2] = (2 * ((float)goal[0]) / {rows}) - 1;
        states[i * statesize + displacement + 3] = (2 * ((float)goal[1]) / {cols}) - 1;
        int bin_idx = (myposs[localctr][0] / {binwidth}) * {binc} + (myposs[localctr][1] / {binwidth});
        int bin_pos = atomicAdd(&(bincounters[bin_idx]), 1);
        myposs[localctr][2] = bin_idx;
        myposs[localctr][3] = bin_pos;
        myposs[localctr][4] = i;
        localctr++;
    }}
    __syncthreads();

    // Perform parallel exlusive scan to accumulate bin sizes so we can index into bins
    for(int i = start; i < {nbins_rounded}; i += step){{
        binpfx[i] = bincounters[i];
    }}
    __syncthreads();
    exclusiveScan({nbins_rounded}, binpfx);

    // Populate bins with the agent positions I am responsible for
    for(int i = 0; i < localctr; i++){{
        int bin_idx = myposs[i][2];
        int bin_pos = myposs[i][3];
        int binsidx = binpfx[bin_idx] + bin_pos;
        bins[binsidx][0] = (unsigned short) myposs[i][0];
        bins[binsidx][1] = (unsigned short) myposs[i][1];
    }}
    __syncthreads();

    // Render rest of state for agent positions I am responsible for
    for(int i = 0; i < localctr; i++){{
        // load my position in field
        int myr = myposs[i][0];
        int myc = myposs[i][1];

        // Determine receptive field overlap with bins
        int TLR = max(0, (myr - view_range) / {binwidth});
        int TLC = max(0, (myc - view_range) / {binwidth});
        int BRR = min({binr - 1}, (myr + view_range) / {binwidth});
        int BRC = min({binc - 1}, (myc + view_range) / {binwidth});

        // Iterate over each agent in all possible bins
        for(int binr = TLR; binr <= BRR; binr++){{
            for(int binc = TLC; binr <= BRC; binc++){{
                int bin_idx = binr * {binc} + binc;
                for(int binsidx = binpfx[bin_idx]; binsidx < binpfx[bin_idx] + bincounters[bin_idx]; binsidx++){{
                    int otherr = (int)(bins[binsidx][0]); // location in field
                    int otherc = (int)(bins[binsidx][1]); 
                    int viewr = otherr - myr + view_range;
                    int viewc = otherc - myc + view_range;
                    // If if statement passes, then other agent is in view range.
                    if(0 <= viewr && viewr < {view_size} && 0 <= viewc && viewc < {view_size}){{ 
                        states[viewr * {view_size} + viewc] = 1.0;
                    }}
                }}
            }}
        }}

        // Render -1 for walls
        for(int viewr = 0; viewr < {view_size}; viewr++){{
            for(int viewc = 0; viewc < {view_size}; viewc++){{
                int fieldr = myr + viewr - view_range;
                int fieldc = myc + viewc - view_range;
                if(fieldr < 0 || fieldc < 0 || fieldr >= {rows} || fieldc >= {cols}){{
                    int stateidx = statesize * i + viewr * {view_size} + viewc;
                    states[stateidx] = -1;
                }}
            }}
        }}
    }}
}}"""
print(kernel)

mod = SourceModule(kernel)
step = mod.get_function("tensor")