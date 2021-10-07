import numpy as np
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num, base, digs):
    """ convert base-10 integer to base-n array of fixed no. of digits 
    return array (of length = digs)"""
    res = np.zeros(digs, dtype=np.int32)
    q = num
    for i in range(digs):
        res[i]=q%base
        q = floor(q/base)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2int(arr, base):
    """ convert array from given base to base-10  --> return integer"""
    res = 0
    for i in range(len(arr)):
        res+=(base**i)*arr[i]
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="[", sep=",", end="]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for i in range(len(arr)):
        res+=str(arr[i])+sep
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=   
def strD(arr, sep="\n", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-=\nDICT: "+caption+"\n=-=-=-=-==-=-=-=-=\n"
    for i in arr:
        res+=str(i) + "\t\t" + str(arr[i]) + sep
    return res + "=-=-=-=-==-=-=-=-=\n"
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strV(SV):
    """ converts state vector to appropiate string for storing in Q-dictionary --> return string """
    if (type(SV)==np.ndarray):
        return strA(SV)
    else:
        return str(SV)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=