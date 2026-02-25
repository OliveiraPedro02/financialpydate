import numpy as np
from financialpydate import FinancialCalendar


Israel_Settlement: FinancialCalendar = FinancialCalendar(holidays=np.array((), dtype='datetime64[D]'))
