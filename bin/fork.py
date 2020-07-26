import numpy as np
import configparser
import sys
import os
import subprocess
from mpi4py import MPI
import stat

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class FORK():

    def __init__(self):
        a = 2 + 1

    def mpi_fork(self, n):
        """Re-launches the current script with workers
        Returns "parent" for original parent, "child" for MPI children
        (from https://github.com/garymcintire/mpi_util/)
        """
        global comm, rank
        if n<=1:
          return "child", comm, rank
        if os.getenv("IN_MPI") is None:
           env = os.environ.copy()
           env.update(
              MKL_NUM_THREADS="1",
              OMP_NUM_THREADS="1",
              IN_MPI="1" 
           )
           print( ["mpirun", "-np  ", str(n), sys.executable] + sys.argv)
           subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
           return "parent", comm, rank
        else:
           #global nworkers, rank
           nworkers = comm.Get_size()
           rank = comm.Get_rank()
           return "child", comm, rank

