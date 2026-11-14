from abc import ABC, abstractmethod
from functools import cache
from typing import overload

import numba
import numpy as np
import numpy.typing as npt

from financialpydate.date_handler import _civil_from_days, _days_from_civil, _is_leap_int
from financialpydate import FinancialCalendar
from financialpydate.financial_calendar import BUSINESS_DAY_TABLE_START, cumulative_business_days

from financialpydate.numpy_types import NumpyDateType


@overload
def equalize_variable_types(
    start_date: npt.NDArray[NumpyDateType], end_date: npt.NDArray[NumpyDateType]
) -> tuple[npt.NDArray[NumpyDateType], npt.NDArray[NumpyDateType]]: ...


@overload
def equalize_variable_types(
    start_date: NumpyDateType, end_date: npt.NDArray[NumpyDateType]
) -> tuple[npt.NDArray[NumpyDateType], npt.NDArray[NumpyDateType]]: ...


@overload
def equalize_variable_types(
    start_date: npt.NDArray[NumpyDateType], end_date: NumpyDateType
) -> tuple[npt.NDArray[NumpyDateType], npt.NDArray[NumpyDateType]]: ...


@overload
def equalize_variable_types(
    start_date: NumpyDateType, end_date: NumpyDateType
) -> tuple[NumpyDateType, NumpyDateType]: ...


def equalize_variable_types(
    start_date: npt.NDArray[NumpyDateType] | NumpyDateType, end_date: npt.NDArray[NumpyDateType] | NumpyDateType
) -> tuple:
    """
    Convert the `start_date` and `end_date` variables to the same type.
    If single type is given the single type will be returns if array and a single type is given then two arrays with
    the shape of the first array will be return.
    If both parameters are arrays then they must have the same shape. Otherwise, an error will be raised.

    The scenarios are:
        - start_date: NumpyDateType and end_date: NumpyDateType. The return will be (NumpyDateType, NumpyDateType)
        - start_date: npt.NDArray[NumpyDateType] and end_date: NumpyDateType. The return will be
        (npt.NDArray[NumpyDateType], npt.NDArray[NumpyDateType])
        - start_date: NumpyDateType and end_date: npt.NDArray[NumpyDateType]. The return will be
        (npt.NDArray[NumpyDateType], npt.NDArray[NumpyDateType])
        - start_date: npt.NDArray[NumpyDateType] and end_date: npt.NDArray[NumpyDateType] with same shape.
        The return will be (npt.NDArray[NumpyDateType], npt.NDArray[NumpyDateType])

    Parameters
    ----------
    start_date: npt.NDArray[NumpyDateType] | NumpyDateType
        start date or array of start dates to be equalized.
    end_date:  npt.NDArray[NumpyDateType] | NumpyDateType
        end date or array of start dates to be equalized.

    Returns
    -------


    """
    # following numpy recomendation. np.isscalar returns true for array with 0 dimension. For example
    # x = np.array(0), np.isscalar(x) will be True instead of False

    is_start_scalar = np.ndim(start_date) == 0
    is_end_scalar = np.ndim(end_date) == 0
    if not is_start_scalar and is_end_scalar:
        return start_date, np.full(fill_value=end_date, shape=start_date.shape, dtype=start_date.dtype)
    elif is_start_scalar and not is_end_scalar:
        return (np.full(fill_value=start_date, shape=end_date.shape, dtype=start_date.dtype), end_date)
    elif not (is_start_scalar and is_end_scalar):
        assert start_date.shape == end_date.shape, 'start dates and end dates must have equal shape.'

    return start_date, end_date


# ---------------------------------------------------------------------------------------------------------------------
# Input/output plumbing shared by the numba kernels.
#
# The kernels work on flat int64 arrays of "days since 1970-01-01" (the integer behind datetime64[D]). Each date is
# decomposed into (year, month, day) exactly once inside the kernel, instead of repeated datetime64 unit conversions.
# ---------------------------------------------------------------------------------------------------------------------


def _as_day_numbers(start_date, end_date) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], tuple[int, ...]]:
    start = np.asarray(start_date, dtype='datetime64[D]')
    end = np.asarray(end_date, dtype='datetime64[D]')
    shape = np.broadcast_shapes(start.shape, end.shape)
    if start.shape != shape:
        start = np.broadcast_to(start, shape)
    if end.shape != shape:
        end = np.broadcast_to(end, shape)
    return (
        np.ascontiguousarray(start).reshape(-1).view(np.int64),
        np.ascontiguousarray(end).reshape(-1).view(np.int64),
        shape,
    )


def _restore_shape(values: npt.NDArray, shape: tuple[int, ...]):
    """Give the kernel output the broadcast input shape; a 0-d result is returned as a numpy scalar."""
    return values.reshape(shape)[()]


@numba.njit(cache=True, inline='always')
def _thirty_360_days(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:
    return 360 * (y2 - y1) + 30 * (m2 - m1) + d2 - d1


@numba.njit(cache=True, inline='always')
def _is_last_day_of_feb_int(y: int, m: int, d: int) -> bool:
    return m == 2 and d == (29 if _is_leap_int(y) else 28)


@numba.njit(cache=True, inline='always')
def _days_in_month(y: int, m: int) -> int:
    if m == 2:
        return 29 if _is_leap_int(y) else 28
    if m == 4 or m == 6 or m == 9 or m == 11:
        return 30
    return 31


@numba.njit(cache=True, inline='always')
def _bond_basis_days(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:
    if d1 >= 30:
        d1 = 30
        if d2 > 30:
            d2 = 30
    return _thirty_360_days(y1, m1, d1, y2, m2, d2)


@numba.njit(cache=True)
def _thirty_360_bond_basis_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        out[i] = _bond_basis_days(y1, m1, d1, y2, m2, d2)
    return out


@numba.njit(cache=True)
def _thirty_360_italian_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        d1 = min(d1, 30)
        d2 = min(d2, 30)
        # any day after the 27th of February counts as the 30th
        if m1 == 2 and d1 > 27:
            d1 = 30
        if m2 == 2 and d2 > 27:
            d2 = 30
        out[i] = _thirty_360_days(y1, m1, d1, y2, m2, d2)
    return out


@numba.njit(cache=True)
def _thirty_360_nasd_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        d1 = min(d1, 30)
        if d2 == 31:
            if d1 >= 30:
                d2 = 30
            else:
                # the 31st counts as the 1st of the following month
                d2 = 1
                m2 += 1
        out[i] = _thirty_360_days(y1, m1, d1, y2, m2, d2)
    return out


@numba.njit(cache=True)
def _simple_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Whole months count as 1/12 of a year; otherwise falls back to 30/360 bond basis."""
    out = np.empty(start.size, np.float64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        if (
            d1 == d2
            or (d1 > d2 and d2 == _days_in_month(y2, m2))  # e.g. 30 Aug -> 28 Feb
            or (d1 < d2 and d1 == _days_in_month(y1, m1))  # e.g. 28 Feb -> 30 Aug
        ):
            out[i] = (y2 - y1) + (m2 - m1) / 12.0
        else:
            out[i] = _bond_basis_days(y1, m1, d1, y2, m2, d2) / 360.0
    return out


@numba.njit(cache=True, inline='always')
def _years_earlier_afb(y: int, m: int, d: int, years: int) -> int:
    """
    The date ``years`` (>= 1) years before (y, m, d), as QuantLib's AFB convention steps back one year at a time
    (29 Feb -> 28 Feb, and 28 Feb landing in a leap year -> 29 Feb). After any number of such steps, 28 and 29 February
    both end up on the last day of February, and every other date keeps its month and day.
    """
    target_year = y - years
    if m == 2 and d >= 28:
        d = 29 if _is_leap_int(target_year) else 28
    return _days_from_civil(target_year, m, d)


@numba.njit(cache=True, inline='always')
def _afb_year_fraction(d1: int, d2: int) -> float:
    """Actual/Actual AFB for ``d1 < d2``: whole years counted back from ``d2``, then the stub over 365 or 366."""
    y1 = _civil_from_days(d1)[0]
    y2, m2, dd2 = _civil_from_days(d2)
    # Stepping back y2 - y1 years lands in d1's year: either on/after d1 (that many whole years) or before it (one less).
    whole_years = y2 - y1
    new_d2 = d2
    if whole_years > 0:
        new_d2 = _years_earlier_afb(y2, m2, dd2, whole_years)
        if new_d2 < d1:
            whole_years -= 1
            new_d2 = d2 if whole_years == 0 else _years_earlier_afb(y2, m2, dd2, whole_years)

    denominator = 365.0
    y_new_d2 = _civil_from_days(new_d2)[0]
    if _is_leap_int(y_new_d2):
        feb_29 = _days_from_civil(y_new_d2, 2, 29)
        if new_d2 > feb_29 and d1 <= feb_29:
            denominator += 1.0
    else:
        if _is_leap_int(y1):
            feb_29 = _days_from_civil(y1, 2, 29)
            if new_d2 > feb_29 and d1 <= feb_29:
                denominator += 1.0

    return whole_years + (new_d2 - d1) / denominator


@numba.njit(cache=True)
def _actual_actual_afb_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    out = np.empty(start.size, np.float64)
    for i in range(start.size):
        s = start[i]
        e = end[i]
        if s == e:
            out[i] = 0.0
        elif s < e:
            out[i] = _afb_year_fraction(s, e)
        else:
            out[i] = -_afb_year_fraction(e, s)
    return out


@numba.njit(cache=True)
def _thirty_e_360_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        out[i] = _thirty_360_days(y1, m1, min(d1, 30), y2, m2, min(d2, 30))
    return out


@numba.njit(cache=True)
def _thirty_e_360_isda_kernel(
    start: npt.NDArray[np.int64], end: npt.NDArray[np.int64], is_end_date_on_termination: bool
) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        if d1 == 31 or _is_last_day_of_feb_int(y1, m1, d1):
            d1 = 30
        if d2 == 31 or (not is_end_date_on_termination and _is_last_day_of_feb_int(y2, m2, d2)):
            d2 = 30
        out[i] = _thirty_360_days(y1, m1, d1, y2, m2, d2)
    return out


@numba.njit(cache=True)
def _thirty_u_360_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        y1, m1, d1 = _civil_from_days(start[i])
        y2, m2, d2 = _civil_from_days(end[i])
        start_is_last_of_feb = _is_last_day_of_feb_int(y1, m1, d1)
        start_is_end_of_month = d1 >= 30 or start_is_last_of_feb
        if (d2 == 31 and start_is_end_of_month) or (start_is_last_of_feb and _is_last_day_of_feb_int(y2, m2, d2)):
            d2 = 30
        if start_is_end_of_month:
            d1 = 30
        out[i] = _thirty_360_days(y1, m1, d1, y2, m2, d2)
    return out


@numba.njit(cache=True)
def _actual_actual_isda_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    out = np.empty(start.size, np.float64)
    for i in range(start.size):
        s = start[i]
        e = end[i]
        if s == e:
            out[i] = 0.0
            continue
        y1 = _civil_from_days(s)[0]
        y2 = _civil_from_days(e)[0]
        days_in_y1 = 366.0 if _is_leap_int(y1) else 365.0
        days_in_y2 = 366.0 if _is_leap_int(y2) else 365.0
        out[i] = (
            (_days_from_civil(y1 + 1, 1, 1) - s) / days_in_y1
            + (e - _days_from_civil(y2, 1, 1)) / days_in_y2
            + (y2 - y1 - 1)
        )
    return out


_NO_LEAP_MONTH_OFFSET = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64)


@numba.njit(cache=True, inline='always')
def _no_leap_serial(days: int) -> int:
    y, m, d = _civil_from_days(days)
    serial = y * 365 + _NO_LEAP_MONTH_OFFSET[m - 1] + d
    if m == 2 and d == 29:
        serial -= 1
    return serial


@numba.njit(cache=True)
def _no_leap_kernel(start: npt.NDArray[np.int64], end: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out = np.empty(start.size, np.int64)
    for i in range(start.size):
        out[i] = _no_leap_serial(end[i]) - _no_leap_serial(start[i])
    return out


@numba.njit(cache=True)
def _business_days_kernel(
    start: npt.NDArray[np.int64], end: npt.NDArray[np.int64], table: npt.NDArray[np.int32], table_start: int
) -> tuple[npt.NDArray[np.int64], bool]:
    """
    Business days via the cumulative table, with ``np.busday_count`` semantics: ``[start, end)`` when
    ``start <= end``, and ``-count((end, start])`` when ``end < start``.
    Returns ``ok=False`` if any date falls outside the table.
    """
    out = np.empty(start.size, np.int64)
    last = table.size - 1
    for i in range(start.size):
        s = start[i] - table_start
        e = end[i] - table_start
        if e < s:
            s += 1
            e += 1
        if s < 0 or e < 0 or s > last or e > last:
            return out, False
        out[i] = table[e] - table[s]
    return out, True


@cache
def _default_business_day_table() -> npt.NDArray[np.int32]:
    """Table for ``np.busday_count``'s default calendar (Monday to Friday, no holidays)."""
    return cumulative_business_days(np.busdaycalendar())


_BUSINESS_DAY_TABLE_START_INT = int(BUSINESS_DAY_TABLE_START.astype(np.int64))


class DayCounter(ABC):
    @property
    @abstractmethod
    def code(self) -> str: ...

    @overload
    def day_count(
        self,
        start_date: npt.NDArray[NumpyDateType],
        end_date: npt.NDArray[NumpyDateType],
        calendar: FinancialCalendar | None = None,
    ) -> npt.NDArray[np.int_]: ...

    @overload
    def day_count(
        self, start_date: npt.NDArray[NumpyDateType], end_date: NumpyDateType, calendar: FinancialCalendar | None = None
    ) -> npt.NDArray[np.int_]: ...

    @overload
    def day_count(
        self, start_date: NumpyDateType, end_date: npt.NDArray[NumpyDateType], calendar: FinancialCalendar | None = None
    ) -> npt.NDArray[np.int_]: ...

    @overload
    def day_count(
        self, start_date: NumpyDateType, end_date: NumpyDateType, calendar: FinancialCalendar | None = None
    ) -> np.int_: ...

    @abstractmethod
    def day_count(self, start_date, end_date, calendar=None):
        """
        Returns the number of days between the given start date and end date.
        Parameters
        ----------
        start_date: NumpyDateType | npt.NDArray[NumpyDateType]
            start date or array of start dates
        end_date: NumpyDateType | npt.NDArray[NumpyDateType]
            start date or array of start dates
        Returns
        -------
        npt.NDArray[np.int_] | int
            the number of days between the given start date and end date.

        """
        ...

    @overload
    def __call__(
        self,
        start_date: npt.NDArray[NumpyDateType],
        end_date: npt.NDArray[NumpyDateType],
        calendar: FinancialCalendar | None = None,
    ) -> npt.NDArray[np.double]: ...

    @overload
    def __call__(
        self, start_date: npt.NDArray[NumpyDateType], end_date: NumpyDateType, calendar: FinancialCalendar | None = None
    ) -> npt.NDArray[np.double]: ...

    @overload
    def __call__(
        self, start_date: NumpyDateType, end_date: npt.NDArray[NumpyDateType], calendar: FinancialCalendar | None = None
    ) -> npt.NDArray[np.double]: ...

    @overload
    def __call__(
        self, start_date: NumpyDateType, end_date: NumpyDateType, calendar: FinancialCalendar | None = None
    ) -> float: ...

    @abstractmethod
    def __call__(self, start_date, end_date, calendar=None) -> npt.NDArray[np.double] | float:
        """
        Returns the year fraction for the given start date and end date.
        Parameters
        ----------
        start_date: NumpyDateType | npt.NDArray[NumpyDateType]
            start date or array of start dates
        end_date: NumpyDateType | npt.NDArray[NumpyDateType]
            start date or array of start dates
        Returns
        -------
        npt.NDArray[np.double] | float
            the year fraction base on the given start date and end date.

        """
        ...

    @property
    def is_additive(self) -> bool:
        return False


class ActualDayCounter(DayCounter):
    def day_count(self, start_date, end_date, *args, **kwargs):
        return (end_date - start_date).astype('timedelta64[D]').astype(int)

    @property
    def is_additive(self) -> bool:
        return True


class Actual360(ActualDayCounter):
    @property
    def code(self):
        return 'ACT/360'

    def __call__(self, start_date, end_date, *args, **kwargs):
        return self.day_count(start_date, end_date) / 360


class Actual365(ActualDayCounter):
    @property
    def code(self):
        return 'ACT/365'

    def __call__(self, start_date, end_date, *args, **kwargs):
        return self.day_count(start_date, end_date) / 365.0


class Actual364(ActualDayCounter):
    @property
    def code(self):
        return 'ACT/364'

    def __call__(self, start_date, end_date, *args, **kwargs):
        return self.day_count(start_date, end_date) / 364.0


class _ActualFixedIncludeLastDay(ActualDayCounter):
    """Actual days over a fixed denominator, optionally counting the end date as well (QuantLib ``includeLastDay``)."""

    __slots__ = 'include_last_day'
    _denominator: float
    _code: str

    def __init__(self, include_last_day: bool = False):
        self.include_last_day: bool = include_last_day

    @property
    def code(self):
        return f'{self._code}(Inc)' if self.include_last_day else self._code

    def day_count(self, start_date, end_date, *args, **kwargs):
        days = super().day_count(start_date, end_date)
        return days + 1 if self.include_last_day else days

    def __call__(self, start_date, end_date, *args, **kwargs):
        return self.day_count(start_date, end_date) / self._denominator

    @property
    def is_additive(self) -> bool:
        # with the last day included, splitting a period counts the split date twice
        return not self.include_last_day


class Actual366(_ActualFixedIncludeLastDay):
    _denominator = 366.0
    _code = 'ACT/366'


class Actual36525(_ActualFixedIncludeLastDay):
    _denominator = 365.25
    _code = 'ACT/365.25'


class Nl365(DayCounter):
    """
    Actual/365 No Leap: actual days, ignoring every 29 February in the period (QuantLib's
    ``Actual365Fixed(Actual365Fixed.NoLeap)``).
    """

    @property
    def code(self):
        return 'NL/365'

    def day_count(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_no_leap_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        return (self.day_count(start_date, end_date)) / 365.0


class Business252(DayCounter):
    @property
    def code(self):
        return '252'

    def day_count(self, start_date, end_date, calendar=None):
        if np.ndim(start_date) == 0 and np.ndim(end_date) == 0:
            # A single pair: numpy's C routine is cheaper than preparing arrays for the table lookup.
            if calendar is None:
                return np.busday_count(start_date, end_date)
            return np.busday_count(start_date, end_date, busdaycal=calendar.numpy_calendar)
        table = _default_business_day_table() if calendar is None else calendar.cumulative_business_days
        start, end, shape = _as_day_numbers(start_date, end_date)
        result, in_table_range = _business_days_kernel(start, end, table, _BUSINESS_DAY_TABLE_START_INT)
        if in_table_range:
            return _restore_shape(result, shape)
        # Dates outside the precomputed table: fall back to numpy's (slower) business-day count.
        if calendar is None:
            return np.busday_count(start_date, end_date)
        return np.busday_count(start_date, end_date, busdaycal=calendar.numpy_calendar)

    def __call__(self, start_date, end_date, calendar=None):
        return self.day_count(start_date, end_date, calendar) / 252


class ActualActual(ActualDayCounter):
    """Actual/Actual (ISDA)."""

    @property
    def code(self):
        return 'ACT/ACT'

    def __call__(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_actual_actual_isda_kernel(start, end), shape)


class ActualActualAFB(ActualDayCounter):
    """
    Actual/Actual (AFB, also called Euro): whole years counted back from the end date, plus the remaining stub over 366
    if it contains a 29 February, otherwise 365.
    """

    @property
    def code(self):
        return 'ACT/ACT AFB'

    def __call__(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_actual_actual_afb_kernel(start, end), shape)

    @property
    def is_additive(self) -> bool:
        return False


class Thirty360(DayCounter):
    @property
    def code(self):
        return '30/360'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/360 convention"""
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_360_bond_basis_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class Thirty365(DayCounter):
    """
    30/365 with the ISO 20022 date adjustment rules (a 31st becomes the 30th, for both dates), matching QuantLib's
    ``Thirty365`` since its September 2025 change (verified against QuantLib 1.41).
    """

    @property
    def code(self):
        return '30/365'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/365 convention"""
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_e_360_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/365 convention"""
        return self.day_count(start_date, end_date) / 365


class ThirtyE360(DayCounter):
    @property
    def code(self):
        return '30E/360'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/360 convention"""
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_e_360_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class ThirtyE360ISDA(DayCounter):
    __slots__ = 'is_end_date_on_termination'

    def __init__(self, is_end_date_on_termination: bool = False):
        self.is_end_date_on_termination: bool = is_end_date_on_termination

    @property
    def code(self):
        return '30E/360ISDA'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """
        Returns number of days between start_date and end_date, using ThirtyE/360 ISDA convention.

        is_end_date_on_termination : whether accrual period end date falls on the termination date.
        """
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(
            _thirty_e_360_isda_kernel(start, end, bool(self.is_end_date_on_termination)),
            shape,
        )

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class ThirtyU360(DayCounter):
    @property
    def code(self):
        return '30U/360'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/360 US convention."""
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_u_360_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class Thirty360German(ThirtyE360ISDA):
    """30/360 German. Same rules as 30E/360 ISDA (QuantLib implements both identically)."""

    @property
    def code(self):
        return '30/360 German'


class Thirty360Italian(DayCounter):
    """30/360 Italian: like 30E/360, and any day after 27 February also counts as the 30th."""

    @property
    def code(self):
        return '30/360 Italian'

    def day_count(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_360_italian_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        return self.day_count(start_date, end_date) / 360


class Thirty360NASD(DayCounter):
    """30/360 NASD: an end date on the 31st counts as the 30th if the start is the 30th/31st, otherwise as the 1st of
    the next month."""

    @property
    def code(self):
        return '30/360 NASD'

    def day_count(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_360_nasd_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        return self.day_count(start_date, end_date) / 360


class SimpleDayCounter(DayCounter):
    """
    Whole-month periods count as an exact number of twelfths of a year (end of month to end of month included);
    anything else falls back to 30/360 bond basis. Day counts are always 30/360 bond basis.
    """

    @property
    def code(self):
        return 'Simple'

    def day_count(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_thirty_360_bond_basis_kernel(start, end), shape)

    def __call__(self, start_date, end_date, *args, **kwargs):
        start, end, shape = _as_day_numbers(start_date, end_date)
        return _restore_shape(_simple_kernel(start, end), shape)


class OneOne(ActualDayCounter):
    @property
    def code(self):
        return '1/1'

    def __call__(self, start_date, end_date, *args, **kwargs):
        return np.ones_like(self.day_count(start_date, end_date), dtype=np.float64)
