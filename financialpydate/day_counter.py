from abc import ABC, abstractmethod
from typing import overload

import numpy as np
import numpy.typing as npt

from financialpydate.date_handler import _is_last_day_of_feb, day, isleap, month, year
from financialpydate import FinancialCalendar

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


class Nl365(DayCounter):
    @property
    def code(self):
        return 'NL/365'

    def day_count(self, start_date, end_date, *args, **kwargs):
        return (end_date - start_date).astype('timedelta64[D]').astype(int) + (
            isleap(year(start_date)) + isleap(year(end_date))
        ) * -1.0

    def __call__(self, start_date, end_date, *args, **kwargs):
        return (self.day_count(start_date, end_date)) / 365.0


class Business252(DayCounter):
    @property
    def code(self):
        return '252'

    def day_count(self, start_date, end_date, calendar=None):
        if FinancialCalendar is None:
            return np.busday_count(start_date, end_date)
        return np.busday_count(start_date, end_date, busdaycal=calendar.numpy_calendar)

    def __call__(self, start_date, end_date, calendar=None):
        return self.day_count(start_date, end_date, calendar) / 252


class ActualActual(ActualDayCounter):
    @property
    def code(self):
        return 'ACT/ACT'

    def __call__(self, start_date, end_date, *args, **kwargs):
        if np.isscalar(start_date) and np.isscalar(end_date):
            if start_date == end_date:
                return 0.0
            start_year = year(start_date)
            end_year = year(end_date)

            year_1_diff = 366 if isleap(start_year) else 365
            year_2_diff = 366 if isleap(end_year) else 365

            return (
                (
                    start_date.astype('datetime64[Y]') + np.timedelta64(1, 'Y') - start_date.astype('datetime64[D]')
                ).astype(int)
                / year_1_diff
                + (end_date.astype('datetime64[D]') - end_date.astype('datetime64[Y]')).astype(int) / year_2_diff
                + (end_year - start_year - 1)
            )

        if np.all(start_date == end_date):
            return np.zeros_like(start_date, dtype=float)

        start_year = year(start_date)
        end_year = year(end_date)

        year_1_diff = np.where(isleap(start_year), 366, 365)
        year_2_diff = np.where(isleap(end_year), 366, 365)

        total_sum = (
            (start_date.astype('datetime64[Y]') + np.timedelta64(1, 'Y') - start_date.astype('datetime64[D]')).astype(
                int
            )
            / year_1_diff
            + (end_date - end_date.astype('datetime64[Y]')).astype(int) / year_2_diff
            + (end_year - start_year - 1)
        )

        return np.where(start_date == end_date, 0.0, total_sum)


class Thirty360(DayCounter):
    @property
    def code(self):
        return '30/360'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/360 convention"""
        if np.isscalar(start_date) and np.isscalar(end_date):
            start_day = day(start_date)
            if start_day < 30:
                d1 = start_day
                d2 = day(end_date)
            else:
                d1 = 30
                d2 = np.minimum(d1, day(end_date))
        else:
            start_date, end_date = equalize_variable_types(start_date, end_date)
            start_day = day(start_date)
            d1 = np.where(start_day < 30, start_day, 30)
            end_day = day(end_date)
            d2 = np.where(d1 == 30, np.minimum(d1, end_day), end_day)

        return 360 * (year(end_date) - year(start_date)) + 30 * (month(end_date) - month(start_date)) + d2 - d1

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class Thirty365(DayCounter):
    @property
    def code(self):
        return '30/365'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/365 convention"""
        if np.isscalar(start_date) and np.isscalar(end_date):
            start_day = day(start_date)
            if start_day < 30:
                d1 = start_day
                d2 = day(end_date)
            else:
                d1 = 30
                d2 = np.minimum(d1, day(end_date))
        else:
            start_date, end_date = equalize_variable_types(start_date, end_date)
            start_day = day(start_date)
            d1 = np.where(start_day < 30, start_day, 30)
            end_day = day(end_date)
            d2 = np.where(d1 == 30, np.minimum(d1, end_day), end_day)

        return 360 * (year(end_date) - year(start_date)) + 30 * (month(end_date) - month(start_date)) + d2 - d1

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 365


class ThirtyE360(DayCounter):
    @property
    def code(self):
        return '30E/360'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """Returns number of days between start_date and end_date, using Thirty/360 convention"""
        start_date, end_date = equalize_variable_types(start_date, end_date)
        d1 = np.where(day(start_date) < 30, day(start_date), 30)
        d2 = np.where(day(end_date) < 30, day(end_date), 30)

        return 360 * (year(end_date) - year(start_date)) + 30 * (month(end_date) - month(start_date)) + d2 - d1

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
        v_start_date, v_end_date = equalize_variable_types(start_date, end_date)
        d1 = np.where((day(v_start_date) == 31) | _is_last_day_of_feb(v_start_date), 30, day(v_start_date))

        mask = (day(v_end_date) == 31) | (_is_last_day_of_feb(v_end_date) & (not self.is_end_date_on_termination))
        d2 = np.where(mask, 30, day(v_end_date))

        return 360 * (year(v_end_date) - year(v_start_date)) + 30 * (month(v_end_date) - month(v_start_date)) + d2 - d1

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class ThirtyU360(DayCounter):
    @property
    def code(self):
        return '30U/360'

    def day_count(self, start_date, end_date, *args, **kwargs):
        """
        Returns number of days between start_date and end_date, using ThirtyE/360 ISDA convention.

        is_end_date_on_termination : whether accrual period end date falls on the termination date.
        """
        v_start_date, v_end_date = equalize_variable_types(start_date, end_date)
        mask_d1_date = day(v_start_date) >= 30
        feb_mask_d1 = _is_last_day_of_feb(v_start_date)
        d1_or_end_feb = mask_d1_date | feb_mask_d1
        d1 = np.where(d1_or_end_feb, 30, day(v_start_date))

        mask = ((day(v_end_date) == 31) & d1_or_end_feb) | (feb_mask_d1 & _is_last_day_of_feb(v_end_date))
        d2 = np.where(mask, 30, day(v_end_date))

        return 360 * (year(v_end_date) - year(v_start_date)) + 30 * (month(v_end_date) - month(v_start_date)) + d2 - d1

    def __call__(self, start_date, end_date, *args, **kwargs):
        """Returns fraction in years between start_date and end_date, using Thirty/360 convention"""
        return self.day_count(start_date, end_date) / 360


class OneOne(ActualDayCounter):
    @property
    def code(self):
        return '1/1'

    def __call__(self, start_date, end_date, *args, **kwargs):
        return np.ones_like(self.day_count(start_date, end_date), dtype=np.float64)
