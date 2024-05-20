import numpy as np
from financial_calendar import FinancialCalendar


WeekendsOnly = FinancialCalendar(holidays=np.array((), dtype='datetime64[D]'))
