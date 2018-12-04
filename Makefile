include ../../env.mk

CPPFLAGS=-fPIC -std=c++11 -Wall -Werror -I../common/include -I/usr/include/python3.6
CFLAGS=-fPIC -std=gnu99 -Wall -Werror

all:	py_keras_pose_net.so fb.so mx.so

dmp_network.o: ../common/src/dmp_network.cpp
	$(GPP) -c -o dmp_network.o ../common/src/dmp_network.cpp $(CPPFLAGS) $(OPT) 

KerasPoseNet_gen.o:	KerasPoseNet_gen.cpp KerasPoseNet_gen.h
	$(GPP) -c -o KerasPoseNet_gen.o KerasPoseNet_gen.cpp $(CPPFLAGS) $(OPT)

py_keras_pose_net.o:	py_keras_pose_net.cpp
	$(GPP) -c -o py_keras_pose_net.o py_keras_pose_net.cpp $(CPPFLAGS) $(OPT)

fb.so:	fb.c
	$(GCC) -shared -o fb.so fb.c $(CFLAGS) $(OPT)

mx.so:	mx.c
	$(GCC) -shared -o mx.so mx.c $(CFLAGS) $(OPT)

py_keras_pose_net.so:	dmp_network.o KerasPoseNet_gen.o py_keras_pose_net.o
	$(GPP) -shared -o py_keras_pose_net.so py_keras_pose_net.o KerasPoseNet_gen.o dmp_network.o -ldmpdv -lboost_python3 -lboost_numpy3

clean:
	rm -f *.so *.o
