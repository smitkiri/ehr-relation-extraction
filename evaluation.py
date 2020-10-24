from typing import Iterable, List
from collections import defaultdict

class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp: int = 0, tn: int = 0,
                 fp: int = 0, fn: int = 0):
        """Initizialize."""
        assert type(tp) == int
        assert type(tn) == int
        assert type(fp) == int
        assert type(fn) == int

        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self) -> float:
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self) -> float:
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1) -> float:
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision() * self.recall())
            den = beta**2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self) -> float:
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)


def entity_report(predicted: List[Iterable[int]], 
               actual: List[Iterable[int]]) -> dict:
    """
    Get the linient and strict precision, recall 
    and F1 score for a single entity

    Parameters
    ----------
    predicted : List[Iterable[int]]
        Predicted ranges.
    actual : List[Iterable[int]]
        Actual ranges.

    Returns
    -------
    dict
        Evaluation metrics.

    """
    predicted = list(predicted)
    actual = list(actual)
    
    predicted.sort(key = lambda x: x[0])
    actual.sort(key = lambda x: [0])
    
    scores = {'lenient': defaultdict(int), 'strict': defaultdict(int)}
    visited = [False] * len(actual)
    
    i = 0   # Iterator for predicted values
    j = 0   # Iterator for actual values
    
    while i < len(predicted) and j < len(actual):
        p_start, p_end = predicted[i]
        a_start, a_end = actual[j]
        
        # Loop actual values until there's an overlap
        # or actual value is ahead of predicted
        while p_start > a_end:
            # If the actual range is not used with any  
            # predicted range, it's false negative
            if not visited[j]:
                scores['strict']['fn'] += 1
                scores['linient']['fn'] += 1
                visited[j] = True

            j += 1
            a_start, a_end = actual[j]
        
        # If the ranges exactly match, it's true positive for strict mode
        if p_start == a_start and p_end == a_end:
            scores['strict']['tp'] += 1
            visited[j] = True
        
        # If the ranges overlap, it's a true positive for linient mode
        if a_start <= p_end and p_start <= a_end:
            scores['linient']['tp'] += 1
            visited[j] = True
        
        else:
            scores['strict']['fp'] += 1
            scores['linient']['fp'] += 1
            
        i += 1
        
    # Get any remaining unpredicted ranges
    while j < len(actual):
        scores['linient']['fn'] += 1
        scores['strict']['fn'] += 1
        j += 1
    
    # Get any remaining predicted ranges
    while i < len(predicted):
        scores['linient']['fp'] += 1
        scores['strict']['fp'] += 1
        i += 1
        
    # Calculate the measures for both modes
    report = {}
    for mode in ('strict', 'linient'):
        measures = Measures(tp = scores[mode]['tp'], 
                            fp = scores[mode]['fp'], 
                            tn = scores[mode]['tn'], 
                            fn = scores[mode]['fn'])
        report[mode] = {}
        report[mode]['precision'] = measures.precision()
        report[mode]['recall'] = measures.recall()
        report[mode]['f1_score'] = measures.f1()
    
    return report
        
    
        