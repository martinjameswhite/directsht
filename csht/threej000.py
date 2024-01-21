#!/usr/bin/env python3
#
import numpy  as np
import ctypes as ct
import os


# Find out where we are.
fullpath = os.path.dirname(__file__) + "/"




class Wigner3j:
    def __init__(self,Nl):
        """
        Class to compute and save/load Wigner 3j symbols.
        :param Nl: int. (One plus) the maximum l for which to compute the 3j's
        """
        self.Nl    = Nl
        self.Nsize = self.get_index(Nl,Nl,Nl)
        # Set up c_double_Array objects for storing three-j symbols.
        self.store = (ct.c_double*self.Nsize)()
        self.mylib =  ct.CDLL(fullpath+"threej000_helper.so")
        self.mylib.make_table(ct.c_int(Nl),ct.byref(self.store))
        #
    def __call__(self,l1,l2,l3):
        """
        Compute the Wigner 3j symbol for integer l's and m1=m2=m3=0.
        :param l1: int. l1
        :param l2: int. l2
        :param l3: int. l3
        :return: float. The Wigner 3j symbol
        """
        return(self.store[self.get_index(l1,l2,l3)])
        #
    def get_index(self,l1,l2,l3):
        """
        Get the index of the Wigner 3j symbol (with m1=m2=m3=0) in the table.
        :param l1: int. l1
        :param l2: int. l2
        :param l3: int. l3
        :return: int. The index of the Wigner 3j symbol in the table.
        """
        # Order ell1, ell2 and ell3 such that j1>=j2>=j3.
        ells = [l1, l2, l3]
        j1 = max(ells)
        j3 = min(ells)
        j2 = l1 + l2 + l3 - j1 - j3
        # Work out the index.
        return(  (j1*(j1+1)*(j1+2))//6 + (j2*(j2+1))//2 + j3 )
        #
