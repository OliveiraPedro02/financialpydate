from typing import overload

import numba
import numpy as np
import numpy.typing as npt

from financialpydate.numpy_types import NumpyDateType, IntArrayType, DateArrayType

Jan = 1
Feb = 2
Mar = 3
Apr = 4
May = 5
Jun = 6
Jul = 7
Aug = 8
Sep = 9
Oct = 10
Nov = 11
Dec = 12


@overload
def isleap(year: int) -> np.bool_: ...


@overload
def isleap(year: IntArrayType) -> npt.NDArray[np.bool_]: ...


@numba.njit(fastmath=True, cache=True)
def isleap(year):
    """Return True for leap years, False for non-leap years."""
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


@numba.njit(cache=True, inline='always')
def _is_leap_int(y: int) -> bool:
    return (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))


@numba.njit(cache=True, inline='always')
def _days_in_month(y: int, m: int) -> int:
    if m == 2:
        return 29 if _is_leap_int(y) else 28
    if m == 4 or m == 6 or m == 9 or m == 11:
        return 30
    return 31


@numba.njit(cache=True, inline='always')
def _civil_from_days(z: int) -> tuple[int, int, int]:
    """
    Convert days since 1970-01-01 into (year, month, day) using integer arithmetic only.

    Howard Hinnant's ``civil_from_days`` algorithm (proleptic Gregorian calendar), the same calendar numpy's
    ``datetime64`` uses. One call replaces the ``astype('datetime64[Y]')`` / ``astype('datetime64[M]')`` round trips.
    """
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    y = yoe + era * 400 + (1 if m <= 2 else 0)
    return y, m, d


@numba.njit(cache=True, inline='always')
def _days_from_civil(y: int, m: int, d: int) -> int:
    """Inverse of `_civil_from_days`: days since 1970-01-01 of the given (year, month, day)."""
    if m <= 2:
        y -= 1
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = m - 3 if m > 2 else m + 9
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


@numba.njit(cache=True)
def _ymd_kernel(days: npt.NDArray[np.int64]) -> tuple[IntArrayType, IntArrayType, IntArrayType]:
    n = days.size
    years = np.empty(n, np.int64)
    months = np.empty(n, np.int64)
    days_of_month = np.empty(n, np.int64)
    for i in range(n):
        years[i], months[i], days_of_month[i] = _civil_from_days(days[i])
    return years, months, days_of_month


@overload
def ymd(date: NumpyDateType) -> tuple[np.int64, np.int64, np.int64]: ...


@overload
def ymd(date: DateArrayType) -> tuple[IntArrayType, IntArrayType, IntArrayType]: ...


def ymd(date):
    """
    Return ``(year, month, day)`` of a date or array of dates, computed in a single pass.

    Prefer this over calling `year`, `month` and `day` separately: each of those converts the dates again.
    """
    days = np.asarray(date, dtype='datetime64[D]')
    shape = days.shape
    y, m, d = _ymd_kernel(np.ascontiguousarray(days).reshape(-1).view(np.int64))
    return y.reshape(shape)[()], m.reshape(shape)[()], d.reshape(shape)[()]


@overload
def day(date: NumpyDateType) -> np.uint64: ...


@overload
def day(date: DateArrayType) -> IntArrayType: ...


def day(date):
    return (date - date.astype('datetime64[M]')).astype(int) + 1


@overload
def month(date: NumpyDateType) -> np.uint64: ...


@overload
def month(date: DateArrayType) -> IntArrayType: ...


def month(date):
    return date.astype('datetime64[M]').astype(int) % 12 + 1


@overload
def year(date: NumpyDateType) -> np.uint64: ...


@overload
def year(date: DateArrayType) -> IntArrayType: ...


def year(date):
    return date.astype('datetime64[Y]').astype(np.int64) + 1970


@overload
def _is_last_day_of_feb(date: NumpyDateType) -> np.bool_: ...


@overload
def _is_last_day_of_feb(date: DateArrayType) -> npt.NDArray[np.bool_]: ...


def _is_last_day_of_feb(date):
    """
    Check the last day of february. February in lep year can have 29 days instead of 28. This function makes sure to
    capture such behavior.
    Parameters
    ----------
    date: NumpyDateType | DateArrayType
        date or dates to be checked.

    Returns
    -------
    np.bool_ | npt.NDArray[np.bool_]
        Returns True if it is the last day of February and False if it is not.

    """
    years, months, days = ymd(date)
    return (months == 2) & (days == 28 + np.where(isleap(years), 1, 0))


def add_month_day(dates: DateArrayType, day: int | np.uint64) -> DateArrayType:
    """
    Move each first-of-month date to ``day`` of its month, capped at the month's last day.

    E.g. ``day=31`` gives 31 Jan, 28 Feb (29 in a leap year), 31 Mar, 30 Apr, ..., 31 Jul, 31 Aug, ...
    """
    if day <= 28:
        return dates + np.timedelta64(day - 1, 'D')
    first_days = np.ascontiguousarray(dates, dtype='datetime64[D]')
    shape = first_days.shape
    out = nb_add_month_day(first_days.reshape(-1).view(np.int64), day)
    return out.view('datetime64[D]').reshape(shape)


@numba.njit(cache=True)
def nb_add_month_day(first_days: npt.NDArray[np.int64], day: int) -> npt.NDArray[np.int64]:
    """Single pass over day numbers: ``first + min(day, days in that month) - 1``."""
    out = np.empty(first_days.size, np.int64)
    for i in range(first_days.size):
        y, m, _ = _civil_from_days(first_days[i])
        out[i] = first_days[i] + min(_days_in_month(y, m), day) - 1
    return out
