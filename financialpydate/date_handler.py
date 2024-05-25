from typing import overload

import numba
import numpy as np
import numpy.typing as npt

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
def isleap(year: npt.NDArray[np.uint]) -> npt.NDArray[np.bool_]: ...


@numba.njit(fastmath=True, cache=True)
def isleap(year):
    """Return True for leap years, False for non-leap years."""
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


@overload
def day(date: np.datetime64) -> np.uint: ...


@overload
def day(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.uint]: ...


def day(date):
    return (date - date.astype('datetime64[M]')).astype(int) + 1


@overload
def month(date: np.datetime64) -> np.uint: ...


@overload
def month(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.uint]: ...


def month(date):
    return date.astype('datetime64[M]').astype(int) % 12 + 1


@overload
def year(date: np.datetime64) -> int: ...


@overload
def year(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.uint]: ...


def year(date):
    return date.astype('datetime64[Y]').astype(int) + 1970


@overload
def _is_last_day_of_feb(date: np.datetime64) -> np.bool_: ...


@overload
def _is_last_day_of_feb(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]: ...


def _is_last_day_of_feb(date):
    """
    Check the last day of february. February in lep year can have 29 days instead of 28. This function makes sure to
    capture such behavior.
    Parameters
    ----------
    date: np.datetime64 | npt.NDArray[np.datetime64]
        date or dates to be checked.

    Returns
    -------
    np.bool_ | npt.NDArray[np.bool_]
        Returns True if it is the last day of February and False if it is not.

    """
    last_of_month = 28 + np.where(isleap(year(date)), 1, 0)
    return (month(date) == 2) & (day(date) == last_of_month)


def add_month_day(dates: npt.NDArray[np.datetime64], day: int | np.uint) -> npt.NDArray[np.datetime64]:
    if day <= 28:
        return dates + np.timedelta64(day - 1, 'D')
    leap_year = isleap(year(dates))
    months = month(dates)
    days = nb_add_month_day(months, leap_year, day)
    return dates + days.astype('timedelta64[D]')


@numba.njit(cache=True)
def nb_add_month_day(
    months: npt.NDArray[np.uint], is_leap_year: npt.NDArray[np.bool_], day: npt.NDArray[np.uint64]
) -> npt.NDArray[np.uint64]:
    max_days = np.minimum(np.array([28, 29, 30, 31], np.int64), day) - 1
    days = np.empty_like(months, np.uint64)
    for i, _month in enumerate(months):
        if _month == 2:
            if is_leap_year[i]:
                days[i] = max_days[1]
            else:
                days[i] = max_days[0]
        else:
            if _month % 2 != 0 and _month < 7 or _month % 2 == 0 and _month > 7:
                days[i] = max_days[3]
            else:
                days[i] = max_days[2]
    return days
