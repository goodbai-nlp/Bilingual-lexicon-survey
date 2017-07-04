
all:
	g++ -c -I3rdparty/ -I3rdparty/boost/include -DEIGEN_NO_DEBUG -m64 -O3 -fopenmp  util.cpp
	g++ -c -I3rdparty/tbb/include -I3rdparty/tclap/include -I3rdparty/ -I3rdparty/boost/include  -DEIGEN_NO_DEBUG -m64 -O3 -fopenmp slice_with_embeddings_para.cpp
	g++ -o slice_with_embeddings slice_with_embeddings_para.o util.o -DEIGEN_NO_DEBUG -m64 -fopenmp -O3 -L3rdparty/tbb/lib/intel64/gcc4.1 -ltbb

clean:
	rm -f *.o
