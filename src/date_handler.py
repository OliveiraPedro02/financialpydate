from functools import singledispatch
from typing import overload

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
def isleap(year: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]: ...


def isleap(year):
    """Return True for leap years, False for non-leap years."""
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


@overload
def day(date: np.datetime64) -> np.int_: ...


@overload
def day(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int_]: ...


def day(date):
    return (date - date.astype('datetime64[M]')).astype(int) + 1


@overload
def month(date: np.datetime64) -> np.int_: ...


@overload
def month(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int_]: ...


def month(date):
    return date.astype('datetime64[M]').astype(int) % 12 + 1


@overload
def year(date: np.datetime64) -> int:
    return date.astype('datetime64[Y]').astype(int) + 1970


@overload
def year(date: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int_]:
    return date.astype('datetime64[Y]').astype(int) + 1970


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


def add_month_day(dates: npt.NDArray[np.datetime64], day: int | np.int_) -> npt.NDArray[np.datetime64]:
    max_days = np.minimum([28, 29, 30, 31], day).astype('timedelta64[D]') - 1
    months = month(dates)
    feb_mask = 2 == months
    not_feb_mask = ~feb_mask
    even_months = months % 2 == 0
    days_31 = (~even_months & (months < 7)) | (even_months & (months > 7))
    days_30 = ~days_31 & not_feb_mask
    leap_year = isleap(year(dates))
    new_dates = dates.astype('datetime64[D]', copy=False)
    new_dates[feb_mask & ~leap_year] += max_days[0]
    new_dates[feb_mask & leap_year] += max_days[1]
    new_dates[days_30] += max_days[2]
    new_dates[days_31] += max_days[3]
    return new_dates


@singledispatch
def is_last_day_of_month(dates):
    raise ValueError(f'dates must be of type numpy.datetime64 or a numpy.array of numpy.datetime64')


@is_last_day_of_month.register(np.datetime64)
def _(dates: np.datetime64) -> bool:
    max_days = np.array([28, 29, 30, 31], dtype='timedelta64[D]')
    months = month(dates)
    feb_mask = 2 == months
    months = month(dates)
    if feb_mask:
        if isleap(year(dates)):
            return day(dates) == max_days[1]
        return day(dates) == max_days[0]

    even_month = months % 2 == 0
    days_31 = (~even_month & (months < 7)) | (even_month & (months > 9))
    if days_31:
        return day(dates) == max_days[3]

    return day(dates) == max_days[2]


@is_last_day_of_month.register(np.ndarray)
def _(dates: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
    max_days = np.array([28, 29, 30, 31], dtype='timedelta64[D]')
    months = month(dates)
    feb_mask = 2 == months
    days = day(dates)
    not_feb_mask = ~feb_mask
    even_months = months % 2 == 0
    days_31 = (~even_months & (months < 7)) | (even_months & (months > 9))
    days_30 = ~days_31 & not_feb_mask
    leap_year = isleap(year(dates))
    is_end_of_dates = np.zeros_like(feb_mask, dtype=np.bool_)
    is_end_of_dates[feb_mask & ~leap_year] = days[feb_mask & ~leap_year] == max_days[0]
    is_end_of_dates[feb_mask & leap_year] = days[feb_mask & leap_year] == max_days[1]
    is_end_of_dates[days_30] = days[days_30] == max_days[2]
    is_end_of_dates[days_31] = days[days_31] == max_days[3]
    return is_end_of_dates
