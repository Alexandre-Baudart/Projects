# Generation of a random list

import random

def random_list(N : int,n : int) -> list :
    """
    function which generates a random list

    param N : int
    param n : int
    return : list
    
    """
    data = [random.randrange(n) for i in range(N)]
    return data

# Swap function

def swap(T,i,j):
    """
    function which allow the swap of two elements in a list
    """
    tmp = T[i]
    T[i] = T[j]
    T[j] = tmp
    return T

# 1/Selection sort 

def selection_sort(T : list) -> list :
    """
    function which returns a sorted list according to selection sort method

    param T : list
    return : list
    
    """
    for i in range(0,len(T)):
        mini = i
        for j in range(i+1,len(T)):
            if T[j] < T[mini] :
                mini = j
        if mini != i :
            swap(T,i,mini)
    return T


# 2/Insertion sort

def insertion_sort(T : list) -> list :
    """
    function which returns a sorted list according to insertion sort method

    param T : list
    return : list
    
    """ 
    for i in range(0,len(T)):
        j = i
        val = T[i]
        while j > 0 and val < T[j-1] :
            T[j] = T[j-1]
            j -= 1
        T[j] = val
    return T

# 3/Bubble sort

def bubble_sort(T : list) -> list :
    """
    function which returns a sorted list according to bubble sort method

    param T : list
    return : list
    
    """ 
    permut = True
    while permut :
        permut = False
        for i in range(0,len(T)-1):
            if T[i] > T[i+1] :
                swap(T,i,i+1)
                permut = True
    return T

# 4/Merge sort

from typing import List

def merge(T1 : List[int], T2 : List[int], T : List[int]) -> None :
    i = 0
    j = 0
    while i + j < len(T) :
        if j == len(T2) or (i < len(T1) and T1[i] < T2[j]) :
            T[i + j] = T1[i]
            i += 1
        else :
            T[i + j] = T2[j]
            j += 1

def merge_sort(T : List[int]) -> list :
    """
    function which returns a sorted list according to merge sort method

    param T : list
    return : list
    
    """ 
    n = len(T) 
    if n < 2 :
        return None
    else :
        milieu = n//2
        T1 = T[:milieu]
        T2 = T[milieu:] 

        merge_sort(T1) 
        merge_sort(T2) 

        merge(T1, T2, T)
    return T

data = [2,3,1,0]


        

    







    


