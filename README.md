# warehouse-robots

Might need to add C, C++, CUDA gitignores

To use Numba:
``` conda activate py36_torch ```

To test kernel performance:
``` sudo env PATH="$PATH" nvprof -s ./timing.py ```

To test CPU <-> GPU correctness:
``` ./test.py ```

Good debugging commands (PDB):
```
    pos, goals, field = gpuenv.state.copy_gpu_data()
    gpustate[0][:-4].astype(np.int32).reshape((11,11))
    cpustate[0][:-4].astype(np.int32).reshape((11,11))
```

Kernel Versions:
 - Step
   - Thread/Block Implementation (archived in old_kernels_4.py)
 - Kernel
   - Thread/Block Implementation with local caching (archived as kernel1 in old_kernels_4.py)
   - Thread/Block Implementation (archived as kernel2 in old_kernels_4.py)
   - Thread/Block Implementation with smart sorting (doing rn)