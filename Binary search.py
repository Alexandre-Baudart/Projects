def binary_search(tab : list, x : int) -> int :
    """
    function which returns the position of a target value (x) within an
    array (tab) according to binary search method

    param tab : list
    param x : int
    return : int

    DU (directions for use) : Array 'tab' shall be sorted before binary_search
    function usage

    """
    left = 0
    right = len(tab) - 1
    while left <=right :
        middle = (left + right)//2
        if tab[middle] < x :
            left = middle + 1
        elif tab[middle] > x :
            right = middle - 1
        else :
            return middle

if __name__ == '__main__' :
    assert binary_search([1,4,6,8],6) == 2
    assert binary_search([5,11,13,32,32],5) == 0

    


            
