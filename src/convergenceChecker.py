from typing import List

def chunchIt(values : List, size : int):
    for x in range(0, len(values), size):
        yield values[x:x+size]


def getDiff(val1 :float, val2 : float):
    return abs(val1 - val2)/val1

def checkConvergence(values : List):
    for x, y in chunchIt(values, 2):
        if getDiff(x,y) < 0.01:
            print(x, y)
            print(getDiff(x,y))

lst =[
1.643839453242095,
3.903455357595709,
4.586568738946223,
5.349987044387339,
5.796930963516547,
6.212663962465314,
6.802354602752415,
6.988465240681916,
7.115294559749335,
7.438165072317435,
7.72347488128238,
7.710102159438416,
7.758409999695352,
7.818412075594921,
7.876086446806369,
7.931815772045722,
7.913036350446803,
7.863233890521216,
7.771175841483945,
7.771175841483945]

checkConvergence(lst)