__global__ void step(float *all_rewards, int *all_actions, int *all_poss, int *all_goals, int *all_fields){{
    const int threadidx = threadIdx.x;
    const int nthreads = blockDim.x;
    const int blockidx = blockIdx.x;

    // pull from correct index in overall arrays
    float *rewards = &(all_rewards[blockidx * {nagents}]);
    int *actions = &(all_actions[blockidx * {nagents}]);
    int *poss = &(all_poss[blockidx * {nagents} * 2]);
    int *goals = &(all_goals[blockidx * {nagents} * 2]);
    int *field = &(all_fields[blockidx * {rows} * {cols}]);

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
}}