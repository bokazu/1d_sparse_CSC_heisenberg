gcc_options = -std=c++17 -Wall --pedantic-errors -DMKL_ILP64  -I"${MKLROOT}/include" -g
l_b = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl


program : main.o get_data.o CSC_sparse_count_mat_elements.o spm.o smp.o szz.o CSC_spin.o CSC_sparse_make_hamiltonian.o vec_init.o sparse_lanczos.o sparse_dgemv.o sparse_eigenvec.o sdz.o gso.o
	g++ -o $@ $^ $(l_b)

# program : main.o get_data.o sparse_count_mat_elements.o  spm.o smp.o szz.o vec_init.o
# 	g++ -o $@ $^ $(l_b)

main.o : main.cpp
	g++ -c $(gcc_options) $< $(l_b)

get_data.o : ./CSC_sparse_make_hamiltonian/get_data.cpp
	g++ -c $(gcc_options) $< $(l_b)

CSC_sparse_count_mat_elements.o : ./CSC_sparse_make_hamiltonian/CSC_sparse_count_mat_elements.cpp
	g++ -c $(gcc_options) $< $(l_b)

CSC_sparse_make_hamiltonian.o : ./CSC_sparse_make_hamiltonian/CSC_sparse_make_hamiltonian.cpp
	g++ -c $(gcc_options) $< $(l_b)

spm.o : ./CSC_sparse_make_hamiltonian/spm.cpp
	g++ -c $(gcc_options) $< $(l_b)

smp.o : ./CSC_sparse_make_hamiltonian/smp.cpp
	g++ -c $(gcc_options) $< $(l_b)

szz.o : ./CSC_sparse_make_hamiltonian/szz.cpp
	g++ -c $(gcc_options) $< $(l_b)

CSC_spin.o : ./CSC_sparse_make_hamiltonian/CSC_spin.cpp
	g++ -c $(gcc_options) $< $(l_b)

sparse_lanczos.o : ./CSC_sparse_lanczos/sparse_lanczos.cpp
	g++ -c $(gcc_options) $< $(l_b)

sparse_dgemv.o : ./CSC_sparse_lanczos/sparse_dgemv.cpp
	g++ -c $(gcc_options) $< $(l_b)

sparse_eigenvec.o : ./CSC_sparse_lanczos/sparse_eigenvec.cpp
	g++ -c $(gcc_options) $< $(l_b)

sdz.o : ./CSC_sparse_lanczos/sdz.cpp
	g++ -c $(gcc_options) $< $(l_b)

gso.o : ./CSC_sparse_lanczos/gso.cpp
	g++ -c $(gcc_options) $< $(l_b)

vec_init.o : vec_init.cpp
	g++ -c $(gcc_options) $< $(l_b)


run : program
	./program

clean:
	rm -f ./program

.PHONY : run clean