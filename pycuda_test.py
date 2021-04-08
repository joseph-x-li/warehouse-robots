import pycuda.autoinit
from pycuda.compiler import SourceModule

# define function
mod = SourceModule("""
                   __global__ void add_them(float *dest, float *a, float *b)
                   {
                        const int i = threadIdx.x;
                        dest[i] = a[i] + b[i];
                        }
                        """)

# translate function to python 
add_them = mod.get_function("add_them")


import numpy

# make 2 sets of 400 random floats
a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

# create a set of 0s
dest = numpy.zeros_like(a)


import pycuda.driver as drv

# replace 0s with results of a + b
add_them(drv.Out(dest), drv.In(a), drv.In(b),
         block=(400,1,1), grid=(1,1))

# should print block of 0s -> (a+b) - (a+b)
print(dest - (a + b))