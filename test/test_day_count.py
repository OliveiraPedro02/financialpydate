import datetime as dt

import numpy as np
import QuantLib as ql
from day_counter import (
    Actual360,
    Actual365,
    ActualActual,
    OneOne,
    Thirty360,
    ThirtyE360,
    ThirtyE360ISDA,
    ThirtyU360,
    Business252,
)

# from update_files.get_holidays import holiday_list_numpy
from calendars.all_calendar import all_calendars


class BaseStructure:
    start_date: dt.date
    end_date: dt.date

    def generate_maturity_tests(self, day_count, quantlib_day_count):
        dates = np.arange(np.datetime64(self.start_date), np.datetime64(self.end_date) + 1)
        ql_dates = [ql.Date(date.day, date.month, date.year) for date in dates.astype(dt.date)]
        maturity = ql_dates[-1]
        ql_values = np.array([quantlib_day_count.yearFraction(date, maturity) for date in ql_dates])
        output = day_count(start_date=dates, end_date=np.datetime64(self.end_date))
        return np.all(np.isclose(output, ql_values))

    def generate_start_date_tests(self, day_count, quantlib_day_count):
        dates = np.arange(np.datetime64(self.start_date), np.datetime64(self.end_date) + 1)
        ql_dates = [ql.Date(date.day, date.month, date.year) for date in dates.astype(dt.date)]
        start_date = ql_dates[0]
        ql_values = np.array([quantlib_day_count.yearFraction(start_date, date) for date in ql_dates])
        output = day_count(start_date=np.datetime64(self.start_date), end_date=dates)
        return np.all(np.isclose(output, ql_values))

    def test_actual_360_maturity(self):
        actual_360 = Actual360()
        assert self.generate_maturity_tests(actual_360, ql.Actual360())

    def test_actual_365_maturity(self):
        actual_365 = Actual365()
        assert self.generate_maturity_tests(actual_365, ql.Actual365Fixed())

    def test_actual_actual_maturity(self):
        act_act = ActualActual()
        assert self.generate_maturity_tests(act_act, ql.ActualActual(ql.ActualActual.ISDA))

    def test_thirty_360_maturity(self):
        thirty_360 = Thirty360()
        assert self.generate_maturity_tests(thirty_360, ql.Thirty360(ql.Thirty360.BondBasis))

    def test_thirty_e_360_maturity(self):
        thirty_e_360 = ThirtyE360()
        assert self.generate_maturity_tests(thirty_e_360, ql.Thirty360(ql.Thirty360.EurobondBasis))

    def test_thirty_u_360_maturity(self):
        thirty_e_360 = ThirtyU360()
        assert self.generate_maturity_tests(thirty_e_360, ql.Thirty360(ql.Thirty360.USA))

    def test_thirty_360_isda_maturity(self):
        thirty_e_360 = ThirtyE360ISDA()
        assert self.generate_maturity_tests(thirty_e_360, ql.Thirty360(ql.Thirty360.ISDA))

    def test_one_one_maturity(self):
        one_one = OneOne()
        assert self.generate_maturity_tests(one_one, ql.OneDayCounter())

    def test_actual_360_start_date(self):
        actual_360 = Actual360()
        assert self.generate_start_date_tests(actual_360, ql.Actual360())

    def test_actual_365_start_date(self):
        actual_365 = Actual365()
        assert self.generate_start_date_tests(actual_365, ql.Actual365Fixed())

    def test_actual_actual_start_date(self):
        act_act = ActualActual()
        assert self.generate_start_date_tests(act_act, ql.ActualActual(ql.ActualActual.ISDA))

    def test_thirty_360_start_date(self):
        thirty_360 = Thirty360()
        assert self.generate_start_date_tests(thirty_360, ql.Thirty360(ql.Thirty360.BondBasis))

    def test_thirty_e_360_start_date(self):
        thirty_e_360 = ThirtyE360()
        assert self.generate_start_date_tests(thirty_e_360, ql.Thirty360(ql.Thirty360.EurobondBasis))

    def test_thirty_360_isda_start_date(self):
        thirty_e_360 = ThirtyE360ISDA()
        assert self.generate_start_date_tests(thirty_e_360, ql.Thirty360(ql.Thirty360.ISDA))

    def test_one_one_start_date(self):
        one_one = OneOne()
        assert self.generate_start_date_tests(one_one, ql.OneDayCounter())

    def test_business_252(self):
        business_252 = Business252(calendar=all_calendars["UnitedStates['GovernmentBond']"])
        assert self.generate_start_date_tests(
            business_252, ql.Business252(ql.UnitedStates(ql.UnitedStates.GovernmentBond))
        )


class TestSameDates(BaseStructure):
    start_date = dt.date(2011, 8, 31)
    end_date = dt.date(2011, 8, 31)


class TestLessThanOneYearDates(BaseStructure):
    start_date = dt.date(2011, 8, 31)
    end_date = dt.date(2012, 2, 29)


class Test20YearsDates(BaseStructure):
    start_date = dt.date(2000, 8, 31)
    end_date = dt.date(2022, 8, 31)
