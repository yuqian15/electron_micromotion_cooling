#Test program to solve problems. 
import numpy as np

class A:
    def __init__(self, variables = {}):
        self.variables = {}
        if ("leng") in variables:
            self.variables["leng"] = variables["leng"]
        else:
            self.variables["leng"] = np.array([10,])


        if ("width") in variables:
            self.variables["width"]= variables["width"]
        else:
            self.variables["width"] = np.array([10,])

test = A({'leng': 1, 'width' : 2})
print(test.variables["width"])
