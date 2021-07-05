# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:47:53 2021

@author: druth
"""

import pickle

def save_obj(obj,fpath):
    print('Saving data to '+fpath)
    with open(fpath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_or_pass_on(fpath_or_obj):
    if type(fpath_or_obj) == str:
        fpath = fpath_or_obj
        print('Loading data from '+fpath)
        with open(fpath, 'rb') as handle:
            obj = pickle.load(handle)
    else:
        obj = fpath_or_obj
    return obj