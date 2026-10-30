import numpy as np
def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Jukes-Cantor corrected genetic distance using Hamming distance.
        For example 1.163.

    """
    mismatches = valid = 0
    for a, b in zip(reference_sequence, distant_sequence):
        if a not in 'ACGT' or b not in 'ACGT':
            continue
        valid += 1
        mismatches += (a != b)
    if valid == 0:
        return 0.0
    p = mismatches / valid
    if p >= 0.75:
        return float('inf')
    return valid * (-0.75 * np.log(1 - 4.0 / 3.0 * p))


def kimura_two_parameter(reference_sequence: str, distant_sequence: str) -> float:
    """The Kimura Two Parameter correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Kimura corrected genetic distance using Hamming distance.
        For example 1.196.

    """
    purines, pyrimidines = set("AG"), set("CT")
    ts = tv = valid = 0
    for a, b in zip(reference_sequence, distant_sequence):
        if a not in 'ACGT' or b not in 'ACGT':
            continue
        valid += 1
        if a != b:
            if (a in purines and b in purines) or (a in pyrimidines and b in pyrimidines):
                ts += 1
            else:
                tv += 1
    if valid == 0:
        return 0.0
    P, Q = ts / valid, tv / valid
    if 1 - 2*P - Q <= 0 or 1 - 2*Q <= 0:
        return float('inf')
    return valid * (-0.5 * np.log(1 - 2*P - Q) - 0.25 * np.log(1 - 2*Q))
