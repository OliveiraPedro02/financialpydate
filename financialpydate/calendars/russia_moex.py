import numpy as np
from financialpydate import FinancialCalendar


Russia_MOEX: FinancialCalendar = FinancialCalendar(holidays=np.array((), dtype='datetime64[D]'))
