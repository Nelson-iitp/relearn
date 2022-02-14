import datetime
from math import floor

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num, base, digs):
    """ convert base-10 integer to base-n array of fixed no. of digits 
    return array (of length = digs)"""
    #res = np.zeros(digs, dtype=np.int32)
    res = [ 0 for _ in range(digs) ]
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
def int2baseA(num, base):
    """ convert base-10 integer to base-n array of fixed no. of digits 
    return array (of length = digs)"""
    #res = np.zeros(digs, dtype=np.int32)
    digs = len(base)
    res = [ 0 for _ in range(digs) ]
    q = num
    for i in range(digs):
        res[i]=q%base[i]
        q = floor(q/base[i])
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2intA(arr, base):
    """ convert array from given base to base-10  --> return integer"""
    res = 0
    for i in range(len(arr)):
        res+=(base[i]**i)*arr[i]
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="[\n\t", sep="\n\t", end="\n]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for a in arr:
        res += (str(a) + sep)
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strD(arr, sep="\n", cep="\t:\t", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-="+sep+"DICT: "+caption+sep+"=-=-=-=-==-=-=-=-="+sep
    for i in arr:
        res+=str(i) + cep + str(arr[i]) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strN(format="%Y_%m_%d_%H_%M_%S"):
    """ formated time stamp """
    return datetime.datetime.strftime(datetime.datetime.now(), format)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def show(x, P = print):
    P(strD(x.__dict__, caption=str(x.__class__)))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



