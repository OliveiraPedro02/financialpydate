import numpy as np
from financial_calendar import FinancialCalendar


Russia_MOEX = FinancialCalendar(holidays=np.array((), dtype='datetime64[D]'))
