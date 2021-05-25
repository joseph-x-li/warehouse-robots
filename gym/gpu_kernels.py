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
    with open("kernels/step.cu", 'r') as f:
        kernel = f.read().format(
            rows=rows,
            cols=cols,
            nagents=nagents,
            nenv=nenv,
            WCOLLISION_REWARD=WCOLLISION_REWARD,
            ROBOT_COLLISION_REWARD=ROBOT_COLLISION_REWARD,
            GOAL_REWARD=GOAL_REWARD,
        )
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

    with open("kernels/tensor.cu", 'r') as f:
        kernel = f.read().format(
            rows=rows, 
            cols=cols, 
            view_size=view_size, 
            nagents=nagents, 
            nenv=nenv,
            binr=binr,
            binc=binc,
            nbins_rounded=nbins_rounded,
            binwidth=binwidth,
        )
    return SourceModule(kernel).get_function("tensor")