from typing import Iterable, List, Union
from collections import defaultdict
from annotations import Entity
from ehr import HealthRecord

TAGS = ('Drug', 'Strength', 'Duration', 'Route', 'Form',
        'ADE', 'Dosage', 'Reason', 'Frequency')


class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp: int = 0, tn: int = 0,
                 fp: int = 0, fn: int = 0):
        """Initialize."""
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
            num = (1 + beta ** 2) * (self.precision() * self.recall())
            den = beta ** 2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self) -> float:
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)


def get_entity_report(predicted: List[Iterable[int]],
                      actual: List[Iterable[int]]) -> dict:
    """
    Get the lenient and strict precision, recall
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

    predicted.sort(key=lambda x: x[0])
    actual.sort(key=lambda x: [0])

    scores = {'lenient': defaultdict(int), 'strict': defaultdict(int)}

    i: int = 0  # Iterator for predicted values
    j: int = 0  # Iterator for actual values

    while i < len(predicted) and j < len(actual):
        p_start, p_end = predicted[i]
        a_start, a_end = actual[j]

        # Loop actual values until there's an overlap
        # or actual value is ahead of predicted
        while p_start > a_end:
            # If the actual range is not used with any  
            # predicted range, it's false negative
            scores['strict']['fn'] += 1
            scores['lenient']['fn'] += 1

            j += 1
            if j < len(actual):
                a_start, a_end = actual[j]
            else:
                break

        # Exact match of ranges
        if p_start == a_start and p_end == a_end:
            scores['strict']['tp'] += 1
        else:
            scores['strict']['fp'] += 1

        # Overlap of ranges
        if a_start <= p_end and p_start <= a_end:
            scores['lenient']['tp'] += 1
            j += 1

            # If multiple predicted entities overlap with the same
            # actual entity, only one is considered
            while a_start <= p_end and p_start <= a_end:
                i += 1
                if i < len(predicted):
                    p_start, p_end = predicted[i]

                    # If we find an exact match with overlapping
                    # entities, we need to correct strict score
                    if p_start == a_start and p_end == a_end:
                        scores['strict']['tp'] += 1
                        scores['strict']['fp'] -= 1
                else:
                    break

        else:
            scores['lenient']['fp'] += 1
            i += 1

    # Get any remaining unpredicted ranges
    while j < len(actual):
        scores['lenient']['fn'] += 1
        scores['strict']['fn'] += 1
        j += 1

    # Get any remaining predicted ranges
    while i < len(predicted):
        scores['lenient']['fp'] += 1
        scores['strict']['fp'] += 1
        i += 1

    # Calculate the measures for both modes
    report = {}
    for mode in ('strict', 'lenient'):
        report[mode] = {}
        report[mode]['tp'] = scores[mode]['tp']
        report[mode]['fp'] = scores[mode]['fp']
        report[mode]['fn'] = scores[mode]['fn']

    return report


# noinspection PyTypeChecker
def get_single_ner_report(predicted: Union[dict, List[Entity]],
                          actual: Union[dict, List[Entity]]) -> dict:
    """
    Get the NER report for a single document.

    Parameters
    ----------
    predicted : Union[dict, List[Entity]]
        Predicted values. If it is of type dict, keys should
        indicate entities and values would be predicted ranges.

    actual : Union[dict, List[Entity]]
        Actual values. If it is of type dict, keys should
        indicate entities and values would be predicted ranges.

    Returns
    -------
    dict
        {'strict': {scores}, 'lenient': {scores}}.

    """

    def list_to_dict(entities: List[Entity]) -> dict:
        """
        Convert list of Entity objects to a dictionary
        mapping each entity type to a list of their
        character ranges
        """
        ranges = defaultdict(list)
        for ent in entities:
            ranges[ent.name].append(ent.range)

        return ranges

    if not isinstance(predicted, dict):
        predicted = list_to_dict(predicted)

    if not isinstance(actual, dict):
        actual = list_to_dict(actual)

    report = {}
    for mode in ('strict', 'lenient'):
        report[mode] = {'tp': 0, 'fp': 0, 'fn': 0}

        for t in TAGS:
            report[mode][t] = {'tp': 0, 'fp': 0, 'fn': 0}

    for ent_type in TAGS:
        ent_values = get_entity_report(predicted[ent_type],
                                       actual[ent_type])

        for mode in ('strict', 'lenient'):
            ent_mode = ent_values[mode]

            report[mode]['tp'] += ent_mode['tp']
            report[mode]['fp'] += ent_mode['fp']
            report[mode]['fn'] += ent_mode['fn']

            report[mode][ent_type] = ent_mode

    for mode in ('strict', 'lenient'):
        measure = Measures(tp=report[mode]['tp'],
                           fp=report[mode]['fp'],
                           fn=report[mode]['fn'])

        report[mode]['precision'] = measure.precision()
        report[mode]['recall'] = measure.recall()
        report[mode]['f1'] = measure.f1()

    return report


# noinspection PyTypeChecker
def ner_report(predicted: List[Union[dict, List[Entity]]],
               actual: List[HealthRecord],
               verbose: int = 0) -> dict:
    """
    Generates NER evaluation report for multiple documents.

    Parameters
    ----------
    predicted : List[Union[dict, List[Entity]]]
        Entity predictions for all documents. If it is of type dict,
        keys should indicate entities and values would be
        predicted ranges.

    actual : List[HealthRecord]
        Actual list of HealthRecord objects for all documents.

    verbose : int, optional
        1 to print the report, 0 otherwise. The default is 0.

    Returns
    -------
    dict
        Strict and lenient scores for .

    """
    assert len(predicted) == len(actual)

    report = {}
    for mode in ('strict', 'lenient'):
        report[mode] = {'tp': 0, 'fp': 0, 'fn': 0,
                        'macro': {'precision': 0, 'recall': 0, 'f1': 0},
                        'micro': {'precision': 0, 'recall': 0, 'f1': 0}}

        for t in TAGS:
            report[mode][t] = {'tp': 0, 'fp': 0, 'fn': 0}

    for i in range(len(predicted)):
        single_report = get_single_ner_report(
            predicted[i], list(actual[i].entities.values()))

        for mode in ('strict', 'lenient'):
            report[mode]['tp'] += single_report[mode]['tp']
            report[mode]['fp'] += single_report[mode]['fp']
            report[mode]['fn'] += single_report[mode]['fn']

            report[mode]['macro']['precision'] += single_report[mode]['precision']
            report[mode]['macro']['recall'] += single_report[mode]['recall']
            report[mode]['macro']['f1'] += single_report[mode]['f1']

            for t in TAGS:
                report[mode][t]['tp'] += single_report[mode][t]['tp']
                report[mode][t]['fp'] += single_report[mode][t]['fp']
                report[mode][t]['fn'] += single_report[mode][t]['fn']

    for mode in ('strict', 'lenient'):
        report[mode]['macro']['precision'] /= len(predicted)
        report[mode]['macro']['recall'] /= len(predicted)
        report[mode]['macro']['f1'] /= len(predicted)

        micro_measures = Measures(tp=report[mode]['tp'],
                                  fp=report[mode]['fp'],
                                  fn=report[mode]['fn'])

        report[mode]['micro']['precision'] = micro_measures.precision()
        report[mode]['micro']['recall'] = micro_measures.recall()
        report[mode]['micro']['f1'] = micro_measures.f1()

        for t in TAGS:
            tag_measures = Measures(tp=report[mode][t]['tp'],
                                    fp=report[mode][t]['fp'],
                                    fn=report[mode][t]['fn'])

            report[mode][t]['precision'] = tag_measures.precision()
            report[mode][t]['recall'] = tag_measures.recall()
            report[mode][t]['f1'] = tag_measures.f1()

    def print_report(mode):
        """
        Print the generated report.
        """
        report_mode = report[mode]
        print('Entity\t\tprecision\trecall\t\tf1 score\n')
        for ent_type in TAGS:
            ent_type_str = ent_type + '\t' * (1 - int(len(ent_type) / 8))
            string = ent_type_str + '\t{p:.2f}\t\t{r:.2f}\t\t{f1:.2f}'
            string = string.format(p=round(report_mode[ent_type]['precision'], 2),
                                   r=round(report_mode[ent_type]['recall'], 2),
                                   f1=round(report_mode[ent_type]['f1'], 2))
            print(string)

        print('\n')
        string = 'micro avg\t{mp:.2f}\t\t{mr:.2f}\t\t{f1:.2f}'
        string = string.format(mp=round(report_mode['micro']['precision'], 2),
                               mr=round(report_mode['micro']['recall'], 2),
                               f1=round(report_mode['micro']['f1'], 2))
        print(string)

        string = 'macro avg\t{mp:.2f}\t\t{mr:.2f}\t\t{f1:.2f}'
        string = string.format(mp=round(report_mode['macro']['precision'], 2),
                               mr=round(report_mode['macro']['recall'], 2),
                               f1=round(report_mode['macro']['f1'], 2))
        print(string)

    if verbose == 1:
        print("========================================================")
        print("\t\t   Mode: Strict")
        print("========================================================")
        print_report('strict')

        print('\n\n')

        print("========================================================")
        print("\t\t  Mode: Lenient")
        print("=======================================================")
        print_report('lenient')

    return report

# =============================================================================
# def main():
#     pre = [[1,2], [3,5], [8,14], [15,19], [24, 36], [37, 62], [102, 103]]
#     act = [[3,5], [7,13], [15,19], [21,32], [67, 69], [78, 100]]
#     print(get_entity_report(pre, act))
#     
# main()
# =============================================================================
