from enum import Enum


class Rule(str, Enum):
    CDS_2015 = 'CDS_2015'
    old_CDS = 'old_CDS'
    CDS = 'CDS'
    ThirdWednesDay = 'ThirdWednesDay'  # Not implemented yet
    Twentieth = 'Twentieth'  # Not implemented yet
    Twentieth_IMM = 'Twentieth_IMM'  # Not implemented yet
    backward = 'backward'
    forward = 'forward'
    zero = 'zero'
