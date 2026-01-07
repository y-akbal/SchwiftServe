import numpy as np
import sklearn as sk
import joblib
## These are all allowed !!!! 
## we may at most allow numba but not now
## because of the complications with JIT compilation and
## serialization
import time

class multiply:
    def __init__(self, name: str, compile: bool = False):
        self.name = name
        self.compile = compile
        
    def predict(self, X):
        return 2*X