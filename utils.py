# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:10:20 2020

@author: Smit
"""

import sys
from pickle import dump, load

def drawProgressBar(current, total, string = '', barLen = 20):
    '''
    Draws a progress bar, like [====>    ] 40%
    
    Parameters
    ------------
    current: int/float
             Current progress
    
    total: int/float
           The total from which the current progress is made
             
    string: str
            Additional details to write along with progress
    
    barLen: int
            Length of progress bar
    '''
    percent = current / total
    arrow = ">"
    if percent == 1:
        arrow = ""
    # Carriage return, returns to the begining of line to owerwrite
    sys.stdout.write("\r")
    sys.stdout.write("Progress: [{:<{}}] {}/{}".format("=" * int(barLen * percent) + arrow, 
                                                         barLen, current, total) + string)
    sys.stdout.flush()


def is_whitespace(char):
    '''
    Checks if the character is a whitespace
    
    Parameters
    --------------
    char: str
          A single character string to check
    '''
    # ord() returns unicode and 0x202F is the unicode for whitespace
    if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
        return True
    else:
        return False


def is_punct(char):
    '''
    Checks if the character is a punctuation
    
    Parameters
    --------------
    char: str
          A single character string to check
    '''
    if char == "." or char == "," or char == "!" or char == "?" or char == '\\':
        return True
    else:
        return False
    

def save_pickle(file, variable):
    '''
    Saves variable as a pickle file
    
    Parameters
    -----------
    file: str
          File name/path in which the variable is to be stored
    
    variable: object
              The variable to be stored in a file
    '''
    if file.split('.')[-1] != "pkl":
        file += ".pkl"
        
    with open(file, 'wb') as f:
        dump(variable, f)
        print("Variable successfully saved in " + file)


def open_pickle(file):
    '''
    Returns the variable after reading it from a pickle file
    
    Parameters
    -----------
    file: str
          File name/path from which variable is to be loaded
    '''
    if file.split('.')[-1] != "pkl":
        file += ".pkl"
    
    with open(file, 'rb') as f:
        return load(f)