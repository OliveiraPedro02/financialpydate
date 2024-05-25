import numpy as np
from financialpydate import FinancialCalendar


NullCalendar: FinancialCalendar = FinancialCalendar(holidays=np.array((), dtype='datetime64[D]'))
