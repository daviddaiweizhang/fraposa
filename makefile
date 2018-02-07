FLAGS = -std=c++11 -g -O2 
# local
LIBS = -lblas -llapack -lgsl -lgslcblas -lm -larmadillo
# csg
# LIBS = -I/net/dumbo/home/daiweiz/usr/include/ -L/net/dumbo/home/daiweiz/usr/lib/x86_64-linux-gnu -lblas -llapack -lgsl -lgslcblas -lm -larmadillo -lhdf5 
# flux
# LIBS = -I/home/daiweiz/gsl/include -L/home/daiweiz/gsl/lib -lblas -llapack -lgsl -lgslcblas -lm -larmadillo


build:
	g++ $(FLAGS) trace.rand.cpp -o trace.rand $(LIBS)

trace.comb : trace.comb.cpp
	g++ $(FLAGS) trace.comb.cpp -o trace.comb $(LIBS)
