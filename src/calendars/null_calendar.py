import numpy as np
from financial_calendar import FinancialCalendar


NullCalendar = FinancialCalendar(holidays=np.array((), dtype='datetime64[D]'))
