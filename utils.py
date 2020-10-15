# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:10:20 2020

@author: Smit
"""

import sys
from pickle import dump, load
from IPython.core.display import display, HTML

TPL_HTML = '<span style = "background-color: {color}; border-radius: 5px;">&nbsp;{content}&nbsp;</span>'

COLORS = {"Drug": "#aa9cfc", "Strength": "#ff9561", 
          "Form": "#7aecec", "Frequency": "#9cc9cc", 
          "Route": "#ffeb80", "Dosage": "#bfe1d9", 
          "Reason": "#e4e7d2", "ADE": "#ff8197", 
          "Duration": "#97c4f5"}

def display_ehr(text, entities):    
    '''
    Highlights EHR records with colors and displays
    them as HTML. Ideal for working with Jupyter Notebooks

    Parameters
    ----------
    text : str
        EHR record to render
    entities : dictionary / list
         A list of Entity objects

    Returns
    -------
    None.

    '''
    ent_ranges = []
    
    if isinstance(entities, dict):
        entities = entities.values()
        
    # Each range list would look like [start_idx, end_idx, ent_type]
    for ent in entities:
        for rng in ent.ranges:
            rng.append(ent.name)
            ent_ranges.append(rng)
    
    # Sort ranges by start index
    ent_ranges.sort(key = lambda x: x[0])
    
    # Final text to render
    render_text = ""
    start_idx = 0
    
    # Display legend
    for ent, col in COLORS.items():
        render_text += TPL_HTML.format(content = ent, color = col)
        render_text += "&nbsp" * 5
    
    render_text += '\n'
    render_text += '--' * 50
    render_text += "\n\n"
    
    # Replace each character range with HTML span template
    for rng in ent_ranges:
        render_text += text[start_idx:rng[0]]
        render_text += TPL_HTML.format(content = text[rng[0]:rng[1]], color = COLORS[rng[2]])
        start_idx = rng[1]
    
    render_text += text[start_idx:]
    render_text = render_text.replace("\n", "<br>")
    
    # Render HTML
    display(HTML(render_text))


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