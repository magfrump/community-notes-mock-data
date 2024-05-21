#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:18:15 2024

@author: magfrump
"""

import pandas as pd
from typing import Optional
from scoring.matrix_factorization.matrix_factorization import MatrixFactorization
import pprint

class ToyModel():
    """A reimplementation of the matrix factorization model.
    
    This Toy Model removes the control flow, utility functions, and other
    deployment features in order to enable mock data tests of the core matrix
    factorization algorithm.
    """
    
    def __init__(self, num_factors: Optional[int] = 2):
        self._fit_values = None
        self._num_factors = num_factors
        
    def load_dataframe(self, ratings: pd.DataFrame):
        self._df = ratings
        
    def get_dataframe(self):
        return self._df
    
    def run_mf(self):
        self._mf = MatrixFactorization(numFactors=self._num_factors)
        self._fit_values = self._mf.run_mf(self._df)
        return True
        
    def report(self):
        pprint.pprint(self._fit_values)
        return self._fit_values