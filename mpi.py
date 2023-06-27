from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time

def f(x):
    return 5*x**3 + 3*x**2 + 4*x + 20

def trapezoidal_rule(x0, xn, n, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_n = n // size
    local_x0 = x0 + rank * (local_n * (xn - x0) / n)
    local_xn = local_x0 + (local_n * (xn - x0) / n)
    
    local_h = (local_xn - local_x0) / local_n
    local_sum = 0.0

    local_x = local_x0 + local_h
    for i in range(1, local_n):
        local_sum += f(local_x)
        local_x += local_h

    total_sum = comm.reduce(local_sum, op=MPI.SUM)
    total_h = comm.reduce(local_h, op=MPI.SUM)

    if rank == 0:
        result = total_h * ((f(x0) + f(xn)) / 2 + total_sum)
        return result

    return None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    x0 = 0
    xn = 1000000
    n = 10000000

    if rank == 0:
        print("Calculating integral using trapzoidal rule...")
        start_time = time.time()

    result = trapezoidal_rule(x0, xn, n, comm)

    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")

        num_processes = comm.Get_size()
        num_processes_list = comm.gather(num_processes, root=0)
        elapsed_time_list = comm.gather(elapsed_time, root=0)

        if rank == 0:
            plt.plot(num_processes_list, elapsed_time_list, marker='o')
            plt.xlabel('Number of Processes')
            plt.ylabel('Elapsed Time (seconds)')
            plt.title('Execution Time vs. Number of Processes')
            plt.grid(True)
            plt.show()

    MPI.Finalize()
