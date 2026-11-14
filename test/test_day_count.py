import datetime as dt

import numpy as np
import QuantLib as ql
import pytest

from financialpydate.day_counter import (
    Actual360,
    Actual365,
    ActualActual,
    OneOne,
    Thirty360,
    ThirtyE360,
    ThirtyE360ISDA,
    ThirtyU360,
    Business252,
    Nl365,
    Thirty365,
)

# from update_files.get_holidays import holiday_list_numpy
from financialpydate.calendars.all_calendar import all_calendars


class BaseStructure:
    start_date: dt.date
    end_date: dt.date

    def generate_maturity_tests(self, day_count, quantlib_day_count, **kwargs):
        dates = np.arange(np.datetime64(self.start_date), np.datetime64(self.end_date) + 1)
        ql_dates = [ql.Date(date.day, date.month, date.year) for date in dates.astype(dt.date)]
        maturity = ql_dates[-1]
        ql_values = np.array([quantlib_day_count.yearFraction(date, maturity) for date in ql_dates])
        output = day_count(start_date=dates, end_date=np.datetime64(self.end_date), **kwargs)
        return np.all(np.isclose(output, ql_values))

    def generate_start_date_tests(self, day_count, quantlib_day_count, **kwargs):
        dates = np.arange(np.datetime64(self.start_date), np.datetime64(self.end_date) + 1)
        ql_dates = [ql.Date(date.day, date.month, date.year) for date in dates.astype(dt.date)]
        start_date = ql_dates[0]
        ql_values = np.array([quantlib_day_count.yearFraction(start_date, date) for date in ql_dates])
        output = day_count(start_date=np.datetime64(self.start_date), end_date=dates, **kwargs)
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

    def test_business_252_maturity(self):
        business_252 = Business252()
        assert self.generate_maturity_tests(
            business_252,
            ql.Business252(ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
            calendar=all_calendars["UnitedStates['GovernmentBond']"],
        )

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

    def test_thirty_u_360_start_date(self):
        thirty_u_360 = ThirtyU360()
        assert self.generate_start_date_tests(thirty_u_360, ql.Thirty360(ql.Thirty360.USA))

    def test_thirty_365_maturity(self):
        assert self.generate_maturity_tests(Thirty365(), ql.Thirty365())

    def test_thirty_365_start_date(self):
        assert self.generate_start_date_tests(Thirty365(), ql.Thirty365())

    def test_nl_365_maturity(self):
        assert self.generate_maturity_tests(Nl365(), ql.Actual365Fixed(ql.Actual365Fixed.NoLeap))

    def test_nl_365_start_date(self):
        assert self.generate_start_date_tests(Nl365(), ql.Actual365Fixed(ql.Actual365Fixed.NoLeap))

    def test_one_one_start_date(self):
        one_one = OneOne()
        assert self.generate_start_date_tests(one_one, ql.OneDayCounter())

    def test_business_252(self):
        business_252 = Business252()
        assert self.generate_start_date_tests(
            business_252,
            ql.Business252(ql.UnitedStates(ql.UnitedStates.GovernmentBond)),
            calendar=all_calendars["UnitedStates['GovernmentBond']"],
        )


@pytest.mark.benchmark()
class TestSameDates(BaseStructure):
    start_date = dt.date(2011, 8, 31)
    end_date = dt.date(2011, 8, 31)


@pytest.mark.benchmark()
class TestLessThanOneYearDates(BaseStructure):
    start_date = dt.date(2011, 8, 31)
    end_date = dt.date(2012, 2, 29)


@pytest.mark.benchmark()
class Test20YearsDates(BaseStructure):
    start_date = dt.date(2000, 8, 31)
    end_date = dt.date(2022, 8, 31)


# ---------------------------------------------------------------------------------------------------------------------
# Hard-coded cases ported from QuantLib's test-suite/daycounters.cpp (master, checked 2026-11-14).
# The 30/360 cases are the ISDA examples (https://www.isda.org/2008/12/22/30-360-day-count-conventions/); the USA ones
# reflect QuantLib's June 2025 "Fix 30/360 USA calculation".
# ---------------------------------------------------------------------------------------------------------------------

_MONTHS = {
    m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)
}


def _d(text: str) -> np.datetime64:
    """'20 Aug 2006' -> numpy date."""
    d, m, y = text.split()
    return np.datetime64(dt.date(int(y), _MONTHS[m], int(d)))


THIRTY_360_COMMON_EXAMPLE_1 = [
    ('20 Aug 2006', '20 Feb 2007', 180),
    ('20 Feb 2007', '20 Aug 2007', 180),
    ('20 Aug 2007', '20 Feb 2008', 180),
    ('20 Feb 2008', '20 Aug 2008', 180),
    ('20 Aug 2008', '20 Feb 2009', 180),
    ('20 Feb 2009', '20 Aug 2009', 180),
]

THIRTY_360_BOND_BASIS = THIRTY_360_COMMON_EXAMPLE_1 + [
    ('31 Aug 2006', '28 Feb 2007', 178),
    ('28 Feb 2007', '31 Aug 2007', 183),
    ('31 Aug 2007', '29 Feb 2008', 179),
    ('29 Feb 2008', '31 Aug 2008', 182),
    ('31 Aug 2008', '28 Feb 2009', 178),
    ('28 Feb 2009', '31 Aug 2009', 183),
    ('31 Jan 2006', '28 Feb 2006', 28),
    ('30 Jan 2006', '28 Feb 2006', 28),
    ('28 Feb 2006', '3 Mar 2006', 5),
    ('14 Feb 2006', '28 Feb 2006', 14),
    ('30 Sep 2006', '31 Oct 2006', 30),
    ('31 Oct 2006', '28 Nov 2006', 28),
    ('31 Aug 2007', '28 Feb 2008', 178),
    ('28 Feb 2008', '28 Aug 2008', 180),
    ('28 Feb 2008', '30 Aug 2008', 182),
    ('28 Feb 2008', '31 Aug 2008', 183),
    ('26 Feb 2007', '28 Feb 2008', 362),
    ('26 Feb 2007', '29 Feb 2008', 363),
    ('29 Feb 2008', '28 Feb 2009', 359),
    ('28 Feb 2008', '30 Mar 2008', 32),
    ('28 Feb 2008', '31 Mar 2008', 33),
]

THIRTY_360_EUROBOND_BASIS = THIRTY_360_COMMON_EXAMPLE_1 + [
    ('28 Feb 2006', '31 Aug 2006', 182),
    ('31 Aug 2006', '28 Feb 2007', 178),
    ('28 Feb 2007', '31 Aug 2007', 182),
    ('31 Aug 2007', '29 Feb 2008', 179),
    ('29 Feb 2008', '31 Aug 2008', 181),
    ('31 Aug 2008', '28 Feb 2009', 178),
    ('28 Feb 2009', '31 Aug 2009', 182),
    ('31 Aug 2009', '28 Feb 2010', 178),
    ('28 Feb 2010', '31 Aug 2010', 182),
    ('31 Aug 2010', '28 Feb 2011', 178),
    ('28 Feb 2011', '31 Aug 2011', 182),
    ('31 Aug 2011', '29 Feb 2012', 179),
    ('31 Jan 2006', '28 Feb 2006', 28),
    ('30 Jan 2006', '28 Feb 2006', 28),
    ('28 Feb 2006', '3 Mar 2006', 5),
    ('14 Feb 2006', '28 Feb 2006', 14),
    ('30 Sep 2006', '31 Oct 2006', 30),
    ('31 Oct 2006', '28 Nov 2006', 28),
    ('31 Aug 2007', '28 Feb 2008', 178),
    ('28 Feb 2008', '28 Aug 2008', 180),
    ('28 Feb 2008', '30 Aug 2008', 182),
    ('28 Feb 2008', '31 Aug 2008', 182),
    ('26 Feb 2007', '28 Feb 2008', 362),
    ('26 Feb 2007', '29 Feb 2008', 363),
    ('29 Feb 2008', '28 Feb 2009', 359),
    ('28 Feb 2008', '30 Mar 2008', 32),
    ('28 Feb 2008', '31 Mar 2008', 32),
]

THIRTY_360_USA = THIRTY_360_COMMON_EXAMPLE_1 + [
    ('31 Aug 2006', '28 Feb 2007', 178),
    ('28 Feb 2007', '31 Aug 2007', 180),
    ('31 Aug 2007', '29 Feb 2008', 179),
    ('29 Feb 2008', '31 Aug 2008', 180),
    ('31 Aug 2008', '28 Feb 2009', 178),
    ('28 Feb 2009', '31 Aug 2009', 180),
    ('31 Jan 2006', '28 Feb 2006', 28),
    ('30 Jan 2006', '28 Feb 2006', 28),
    ('28 Feb 2006', '3 Mar 2006', 3),
    ('14 Feb 2006', '28 Feb 2006', 14),
    ('30 Sep 2006', '31 Oct 2006', 30),
    ('31 Oct 2006', '28 Nov 2006', 28),
    ('31 Aug 2007', '28 Feb 2008', 178),
    ('28 Feb 2008', '28 Aug 2008', 180),
    ('28 Feb 2008', '30 Aug 2008', 182),
    ('28 Feb 2008', '31 Aug 2008', 183),
    ('26 Feb 2007', '28 Feb 2008', 362),
    ('26 Feb 2007', '29 Feb 2008', 363),
    ('29 Feb 2008', '28 Feb 2009', 360),
    ('28 Feb 2008', '30 Mar 2008', 32),
    ('28 Feb 2008', '31 Mar 2008', 33),
]

# (start, end, expected, termination date). QuantLib takes the termination date; this package takes
# `is_end_date_on_termination`, i.e. whether `end == termination`.
THIRTY_360_ISDA = (
    [(s, e, n, '20 Aug 2009') for s, e, n in THIRTY_360_COMMON_EXAMPLE_1]
    + [
        (s, e, n, '29 Feb 2012')
        for s, e, n in [
            ('28 Feb 2006', '31 Aug 2006', 180),
            ('31 Aug 2006', '28 Feb 2007', 180),
            ('28 Feb 2007', '31 Aug 2007', 180),
            ('31 Aug 2007', '29 Feb 2008', 180),
            ('29 Feb 2008', '31 Aug 2008', 180),
            ('31 Aug 2008', '28 Feb 2009', 180),
            ('28 Feb 2009', '31 Aug 2009', 180),
            ('31 Aug 2009', '28 Feb 2010', 180),
            ('28 Feb 2010', '31 Aug 2010', 180),
            ('31 Aug 2010', '28 Feb 2011', 180),
            ('28 Feb 2011', '31 Aug 2011', 180),
            ('31 Aug 2011', '29 Feb 2012', 179),
        ]
    ]
    + [
        (s, e, n, '29 Feb 2008')
        for s, e, n in [
            ('31 Jan 2006', '28 Feb 2006', 30),
            ('30 Jan 2006', '28 Feb 2006', 30),
            ('28 Feb 2006', '3 Mar 2006', 3),
            ('14 Feb 2006', '28 Feb 2006', 16),
            ('30 Sep 2006', '31 Oct 2006', 30),
            ('31 Oct 2006', '28 Nov 2006', 28),
            ('31 Aug 2007', '28 Feb 2008', 178),
            ('28 Feb 2008', '28 Aug 2008', 180),
            ('28 Feb 2008', '30 Aug 2008', 182),
            ('28 Feb 2008', '31 Aug 2008', 182),
            ('28 Feb 2007', '28 Feb 2008', 358),
            ('28 Feb 2007', '29 Feb 2008', 359),
            ('29 Feb 2008', '28 Feb 2009', 360),
            ('29 Feb 2008', '30 Mar 2008', 30),
            ('29 Feb 2008', '31 Mar 2008', 30),
        ]
    ]
)

THIRTY_365 = [
    ('17 Jun 2011', '30 Dec 2012', 553),
    ('31 Mar 2025', '30 Apr 2025', 30),
    ('30 Sep 2024', '31 Mar 2025', 180),
    ('30 Mar 2025', '31 Mar 2025', 0),
]

ACTUAL_ACTUAL_ISDA = [
    ('1 Nov 2003', '1 May 2004', 0.497724380567),
    ('1 Feb 1999', '1 Jul 1999', 0.410958904110),
    ('1 Jul 1999', '1 Jul 2000', 1.001377348600),
    ('15 Aug 2002', '15 Jul 2003', 0.915068493151),
    ('15 Jul 2003', '15 Jan 2004', 0.504004790778),
    ('30 Jul 1999', '30 Jan 2000', 0.503892506924),
    ('30 Jan 2000', '30 Jun 2000', 0.415300546448),
]

BUSINESS_252_BRAZIL_DATES = [
    '1 Feb 2002', '4 Feb 2002', '16 May 2003', '17 Dec 2003', '17 Dec 2004', '19 Dec 2005', '2 Jan 2006',
    '13 Mar 2006', '15 May 2006', '17 Mar 2006', '15 May 2006', '26 Jul 2006', '28 Jun 2007', '16 Sep 2009',
    '26 Jul 2016',
]  # fmt: skip
BUSINESS_252_BRAZIL_EXPECTED = [
    0.0039682539683, 1.2738095238095, 0.6031746031746, 0.9960317460317, 1.0000000000000, 0.0396825396825,
    0.1904761904762, 0.1666666666667, -0.1507936507937, 0.1507936507937, 0.2023809523810, 0.912698412698,
    2.214285714286, 6.84126984127,
]  # fmt: skip


def _check_day_counts(day_counter, cases):
    starts = np.array([_d(s) for s, _, _ in cases])
    ends = np.array([_d(e) for _, e, _ in cases])
    expected = np.array([n for _, _, n in cases])
    np.testing.assert_array_equal(day_counter.day_count(starts, ends), expected)
    for start, end, n in zip(starts, ends, expected):
        assert day_counter.day_count(start, end) == n, (start, end)


@pytest.mark.parametrize(
    ('day_counter', 'cases'),
    [
        (Thirty360(), THIRTY_360_BOND_BASIS),
        (ThirtyE360(), THIRTY_360_EUROBOND_BASIS),
        (ThirtyU360(), THIRTY_360_USA),
        (Thirty365(), THIRTY_365),
    ],
    ids=['bond_basis', 'eurobond_basis', 'usa', 'thirty_365'],
)
def test_quantlib_thirty_360_cases(day_counter, cases):
    _check_day_counts(day_counter, cases)


def test_quantlib_thirty_365_year_fraction():
    starts = np.array([_d(s) for s, _, _ in THIRTY_365])
    ends = np.array([_d(e) for _, e, _ in THIRTY_365])
    np.testing.assert_allclose(Thirty365()(starts, ends), [n / 365.0 for _, _, n in THIRTY_365], rtol=0, atol=1e-12)


@pytest.mark.parametrize(('start', 'end', 'expected', 'termination'), THIRTY_360_ISDA)
def test_quantlib_thirty_360_isda_cases(start, end, expected, termination):
    day_counter = ThirtyE360ISDA(is_end_date_on_termination=_d(end) == _d(termination))
    assert day_counter.day_count(_d(start), _d(end)) == expected


def test_quantlib_actual_actual_isda_cases():
    starts = np.array([_d(s) for s, _, _ in ACTUAL_ACTUAL_ISDA])
    ends = np.array([_d(e) for _, e, _ in ACTUAL_ACTUAL_ISDA])
    expected = np.array([x for _, _, x in ACTUAL_ACTUAL_ISDA])
    np.testing.assert_allclose(ActualActual()(starts, ends), expected, rtol=0, atol=1e-10)


def test_quantlib_business_252_brazil_cases():
    """Includes a backwards period (15 May 2006 -> 17 Mar 2006), which must give a negative fraction."""
    dates = np.array([_d(x) for x in BUSINESS_252_BRAZIL_DATES])
    calendar = all_calendars["Brazil['Settlement']"]
    calculated = Business252()(dates[:-1], dates[1:], calendar=calendar)
    np.testing.assert_allclose(calculated, BUSINESS_252_BRAZIL_EXPECTED, rtol=0, atol=1e-12)


def test_business_252_outside_cached_range_falls_back_to_numpy():
    calendar = all_calendars["Brazil['Settlement']"]
    starts = np.array(['1850-01-01', '2150-06-01', '2026-01-02'], dtype='datetime64[D]')
    ends = np.array(['1900-01-01', '2300-06-01', '2025-01-02'], dtype='datetime64[D]')
    expected = np.busday_count(starts, ends, busdaycal=calendar.numpy_calendar)
    np.testing.assert_array_equal(Business252().day_count(starts, ends, calendar=calendar), expected)


def test_nl_365_ignores_29_february():
    nl_365 = Nl365()
    # no 29 February between the dates: same as actual
    assert nl_365.day_count(np.datetime64('2024-03-01'), np.datetime64('2024-12-31')) == 305
    # one 29 February inside the period is skipped
    assert nl_365.day_count(np.datetime64('2024-02-01'), np.datetime64('2024-03-01')) == 28
    assert nl_365(np.datetime64('2023-06-30'), np.datetime64('2024-06-30')) == 1.0


def test_scalar_inputs_return_scalars_and_arrays_keep_shape():
    start = np.datetime64('2020-01-31')
    ends = np.array([['2020-02-29', '2020-03-31'], ['2021-02-28', '2021-08-31']], dtype='datetime64[D]')
    for day_counter in (Thirty360(), ThirtyE360(), ThirtyU360(), ThirtyE360ISDA(), Thirty365(), Nl365()):
        assert np.ndim(day_counter.day_count(start, ends[0, 0])) == 0
        assert day_counter.day_count(start, ends).shape == (2, 2)
    assert np.ndim(ActualActual()(start, ends[0, 0])) == 0
    assert ActualActual()(start, ends).shape == (2, 2)
