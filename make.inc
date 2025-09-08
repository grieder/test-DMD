GSL_DIR=/usr/include/gsl/
IFLAGS=-I${GSL_DIR}/include
CC=mpicxx -std=c++11 -O2 -g 
GSL_LIBS=-L${GSL_DIR}/lib -lgsl -lgslcblas
LAPACK_LIBS=-lblas -llapack
SCALAPACK_LIBS=-lscalapack
LDLIBS=${GSL_LIBS} ${LAPACK_LIBS} ${SCALAPACK_LIBS}
LDFLAGS = -Wl,-rpath,${GSL_DIR}/lib

