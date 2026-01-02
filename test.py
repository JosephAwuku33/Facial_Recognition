# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 15:12:42 2025
This file makes use of pickle to load serialized data from 'encodings.pickle'.
@author: DELL
"""

import pickle

file_path = 'encodings.pickle'

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    print("Data loaded successfully:")
    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
