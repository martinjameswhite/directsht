CC  = gcc
OPT = -O -funroll-loops -fopenmp -fPIC -shared

all: sht_helper threej000_helper


sht_helper: sht_helper.c
	$(CC) $(OPT) -o sht_helper.so sht_helper.c -lm

threej000_helper: threej000_helper.c
	$(CC) $(OPT) -o threej000_helper.so threej000_helper.c -lm


.PHONY: clean
clean:
	rm sht_helper.so threej000_helper.so
