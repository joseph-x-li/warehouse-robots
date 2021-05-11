import pycuda.autoinit
from pycuda.compiler import SourceModule

# each environment gets a block, all threads in block run in parallel to compute step. Launch multiple blocks in parallel
def stepkernel(
    rows,
    cols,
    nagents,
    nenv,
    WCOLLISION_REWARD,
    ROBOT_COLLISION_REWARD,
    GOAL_REWARD,
):
    kernel = f""" \
__global__ void step(float *all_rewards, int *all_actions, int *all_poss, int *all_goals, int *all_fields){{
    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;

    // pull from correct index in overall arrays
    float *rewards = &(all_rewards[blockidx * {nagents}]);
    int *actions = &(all_actions[blockidx * {nagents}]);
    int *poss = &(all_poss[blockidx * {nagents * 2}]);
    int *goals = &(all_goals[blockidx * {nagents * 2}]);
    int *field = &(all_fields[blockidx * {rows * cols}]);

    const int movlookup_r[4] = {{ -1, 0, 1,  0 }};
    const int movlookup_c[4] = {{  0, 1, 0, -1 }};

    int oldposctr = 0;
    int oldpos_r[1000];
    int oldpos_c[1000];

    for(int i = threadidx; i < {nagents}; i += nthreads){{
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
                    reward += {WCOLLISION_REWARD};
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


def tensorkernel(rows, cols, view_size, nagents, nenv):
    from math import ceil
    def findNextPowerOf2(n):
        n = n - 1
        while n & n - 1: n = n & n - 1 
        return n << 1
    MAX_SH = 43_000
    usage = None
    binwidth = 1
    while True:
        binr = ceil(rows / binwidth)
        binc = ceil(cols / binwidth)
        nbins = binr * binc
        nbins_rounded = findNextPowerOf2(nbins)
        usage = nbins_rounded * 8 + nagents * 4
        if usage < MAX_SH:
            break
        binwidth += 1
    
    print(f"nbins_rounded: {nbins_rounded}; uses {usage} bytes")

    kernel = f""" \
__global__ void tensor(float *all_states, int *all_poss, int *all_goals, int *all_fields){{
    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;

    const int view_range = {view_size // 2};
    const int displacement = {view_size ** 2};
    const int statesize = {view_size ** 2 + 4};

    float *states = &(all_states[blockidx * {nagents} * statesize]);
    int *poss = &(all_poss[blockidx * {nagents * 2}]);
    int *goals = &(all_goals[blockidx * {nagents * 2}]);
    int *field = &(all_fields[blockidx * {rows * cols}]);


    for(int i = threadidx; i < {nagents}; i += nthreads){{
        int pos[2];
        int goal[2];
        pos[0] = poss[i * 2];
        pos[1] = poss[(i * 2) + 1];
        goal[0] = goals[i * 2];
        goal[1] = goals[(i * 2) + 1];
        
        states[i * statesize + displacement] = (2 * ((float)pos[0]) / {rows}) - 1;
        states[i * statesize + displacement + 1] = (2 * ((float)pos[1]) / {cols}) - 1;
        states[i * statesize + displacement + 2] = (2 * ((float)goal[0]) / {rows}) - 1;
        states[i * statesize + displacement + 3] = (2 * ((float)goal[1]) / {cols}) - 1;
        
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
                int stateidx = statesize * i + r * {view_size} + c;
                states[stateidx] = fillval;
            }}
        }}
    }}
}}"""

    kernel2 = f""" \
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
    int myposs[50][5];  // positions I am responsible for [0, 1] = r, c; [2, 3] = bin_idx, bin_pos; [4] = index in poss
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
        // load my position in field and position in state vector
        int myr = myposs[i][0];
        int myc = myposs[i][1];
        int myidx = myposs[i][4];

        // Render -1 for walls and 0 for ground.
        for(int viewr = 0; viewr < {view_size}; viewr++){{
            for(int viewc = 0; viewc < {view_size}; viewc++){{
                int fieldr = myr + viewr - view_range;
                int fieldc = myc + viewc - view_range;
                float fillval = 0.0;
                if(fieldr < 0 || fieldc < 0 || fieldr >= {rows} || fieldc >= {cols})
                    fillval = -1.0;
                int stateidx = myidx * statesize + viewr * {view_size} + viewc;
                states[stateidx] = fillval;
            }}
        }}

        // Determine receptive field overlap with bins
        int TLR = max(0, (myr - view_range) / {binwidth});
        int TLC = max(0, (myc - view_range) / {binwidth});
        int BRR = min({binr - 1}, (myr + view_range) / {binwidth});
        int BRC = min({binc - 1}, (myc + view_range) / {binwidth});

        // Iterate over each agent in all possible bins
        for(int binr = TLR; binr <= BRR; binr++){{
            for(int binc = TLC; binc <= BRC; binc++){{
                int bin_idx = binr * {binc} + binc;
                int left = binpfx[bin_idx];
                int right = binpfx[bin_idx] + bincounters[bin_idx];
                for(int binsidx = left; binsidx < right; binsidx++){{
                    int otherr = (int)(bins[binsidx][0]); // location in field
                    int otherc = (int)(bins[binsidx][1]); 
                    int viewr = otherr - myr + view_range;
                    int viewc = otherc - myc + view_range;
                    // If if statement passes, then other agent is in view range.
                    if(0 <= viewr && viewr < {view_size} && 0 <= viewc && viewc < {view_size}){{ 
                        int stateidx = myidx * statesize + viewr * {view_size} + viewc;
                        states[stateidx] = 1.0;
                    }}
                }}
                
            }}
        }}
    }}
    
}}"""

    return SourceModule(kernel2).get_function("tensor")
