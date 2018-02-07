FLAGS = -std=c++11 -g -O2 
-std=c++11 -g ../../research/${name}.cpp -o ${name} -O2 

# Local
LIBS = -lblas -llapack -lgsl -lgslcblas -lm -larmadillo
# FLUX
# LIBS = -I/home/daiweiz/gsl/include -L/home/daiweiz/gsl/lib -lblas -llapack -lgsl -lgslcblas -lm -larmadillo
# CSG
# LIBS = -I/net/dumbo/home/daiweiz/usr/include/ -L/net/dumbo/home/daiweiz/usr/lib/x86_64-linux-gnu -lblas -llapack -lgsl -lgslcblas -lm -larmadillo -lhdf5 


build:
	g++ $(FLAGS) trace.rand.cpp -o trace.rand $(LIBS)

trace.comb : trace.comb.cpp
	g++ $(FLAGS) trace.comb.cpp -o trace.comb $(LIBS)
