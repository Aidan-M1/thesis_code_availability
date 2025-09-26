"""
f1_score_custom.py

Helper function for prompts.py

Author: Aidan Murray
Date: 2025-09-26
"""

def f1_score(y_true, y_pred):
    "Converts the inputs to sets and returns the f1 score of these sets"

    set_1 = set(y_true)
    set_2 = set(y_pred)
    f1 = (2 * len(set_1.intersection(set_2))) / (len(set_1) + len(set_2))
    
    return f1