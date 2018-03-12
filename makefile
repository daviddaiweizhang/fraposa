FLAGS = -std=c++11 -g -O2 
LIBS_LOCAL = -lblas -llapack -lgsl -lgslcblas -lm -larmadillo
LIBS_CSG = -I/net/dumbo/home/daiweiz/usr/include/ -L/net/dumbo/home/daiweiz/usr/lib/x86_64-linux-gnu -lblas -llapack -lgsl -lgslcblas -lm -larmadillo -lhdf5 
LIBS_FLUX = -I/home/daiweiz/gsl/include -L/home/daiweiz/gsl/lib -lblas -llapack -lgsl -lgslcblas -lm -larmadillo

all: trace.comb.o trace.rand.o
.PHONY: all

trace.comb.o : trace.comb.cpp
	@if [ `hostname` = "xps-arch"]; then \
	    echo "Compiling with Local libs"; \
	    g++ $(FLAGS) trace.comb.cpp -o trace.comb.o $(LIBS_LOCAL); \
	elif [ `hostname` = "fantasia" ]; then \
	    echo "Compiling with CSG libs"; \
			export LD_LIBRARY_PATH=/home/daiweiz/gsl/lib:$LD_LIBRARY_PATH; \
			echo ${LD_LIBRARY}; \
	    g++ $(FLAGS) trace.comb.cpp -o trace.comb.o $(LIBS_CSG); \
	else \
	    echo "Compiling with Flux libs"; \
	    g++ $(FLAGS) trace.comb.cpp -o trace.comb.o $(LIBS_FLUX); \
	fi


trace.rand.o : trace.rand.cpp
	@if [ `hostname` = "xps-arch" ]; then \
	    echo "Compiling with Local libs"; \
	    g++ $(FLAGS) trace.rand.cpp -o trace.rand.o $(LIBS_LOCAL); \
	elif [ `hostname` = "fantasia" ]; then \
	    echo "Compiling with CSG libs"; \
			export LD_LIBRARY_PATH=/home/daiweiz/gsl/lib:$LD_LIBRARY_PATH; \
			echo ${LD_LIBRARY}; \
	    g++ $(FLAGS) trace.rand.cpp -o trace.rand.o $(LIBS_CSG); \
	else \
	    echo "Compiling with Flux libs"; \
	    g++ $(FLAGS) trace.rand.cpp -o trace.rand.o $(LIBS_FLUX); \
	fi
