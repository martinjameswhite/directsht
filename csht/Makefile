CC  = gcc
OPT = -O -funroll-loops -fopenmp -fPIC -shared

sht_helper: sht_helper.c
	$(CC) $(OPT) -o sht_helper.so sht_helper.c -lm
