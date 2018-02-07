FLAGS = -std=c++11 -g -O2 
LIBS_LOCAL = -lblas -llapack -lgsl -lgslcblas -lm -larmadillo
LIBS_CSG = -I/net/dumbo/home/daiweiz/usr/include/ -L/net/dumbo/home/daiweiz/usr/lib/x86_64-linux-gnu -lblas -llapack -lgsl -lgslcblas -lm -larmadillo -lhdf5 
LIBS_FLUX = -I/home/daiweiz/gsl/include -L/home/daiweiz/gsl/lib -lblas -llapack -lgsl -lgslcblas -lm -larmadillo


build:
	@if [ `hostname` = "david-XPS-13-9343" ]; then \
	    echo "Local libs"; \
	    g++ $(FLAGS) trace.rand.cpp -o trace.rand $(LIBS_LOCAL); \
	elif [ `hostname` = "fantasia" ]; then \
	    echo "CSG libs"; \
	    g++ $(FLAGS) trace.rand.cpp -o trace.rand $(LIBS_CSG); \
	else \
	    echo "Flux libs"; \
	    g++ $(FLAGS) trace.rand.cpp -o trace.rand $(LIBS_FLUX); \
	fi

trace.comb : trace.comb.cpp
	g++ $(FLAGS) trace.comb.cpp -o trace.comb $(LIBS)
