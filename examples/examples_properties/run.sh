#mpiexec -n 4 python qmain_real.py
mpiexec -n 2 python main_analyze0_ep.py
#mpiexec -n 2 python main_qfix.py
#mpiexec -n 4 python -m memory_profiler main_real.py
#mpiexec -n 4 kernprof -l -v main_real.py
#mpiexec -n 4 kernprof -l -v main_qbasic.py
#mpiexec -n 4 python main_qbasic.py
#kernprof -l -v main_qbasic.py
