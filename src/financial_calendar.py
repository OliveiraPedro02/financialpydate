from typing import overload

import numpy as np
import numpy.typing as npt

from date_handler import day, add_month_day, month
from rule import Rule
from update_files.convention import Convention

CDS_2015_BASE_MONTHS = np.array([0, 3, 6, 9, 12])
DELTAS = [3, 2, 1]


def previous_twentieth(date: np.datetime64, rule: Rule) -> np.datetime64:
    month_date = date.astype('datetime64[M]')
    result = month_date + np.timedelta64(19, 'D')
    if result > date:
        result = (month_date - np.timedelta64(1, 'M')) + np.timedelta64(19, 'D')

    if rule in [Rule.CDS_2015, Rule.old_CDS, Rule.CDS, Rule.Twentieth_IMM]:
        skip_months = month(result) % 3
        if skip_months != 0:
            result = (result.astype('datetime64[M]') - np.timedelta64(skip_months, 'M')) + np.timedelta64(19, 'D')

    return result


def next_twentieth(date: np.datetime64, rule: Rule) -> np.datetime64:
    month_date = date.astype('datetime64[M]')
    result = month_date + np.timedelta64(19, 'D')
    if result < date:
        result = (month_date + np.timedelta64(1, 'M')) + np.timedelta64(19, 'D')

    if rule in [Rule.CDS_2015, Rule.old_CDS, Rule.CDS, Rule.Twentieth_IMM]:
        skip_months = 3 - month(result) % 3
        if skip_months != 3:
            result = (result.astype('datetime64[M]') + np.timedelta64(skip_months, 'M')) + np.timedelta64(19, 'D')

    return result


class FinancialCalendar:
    __sluts__ = ('_stub_days_old_cds', '_calendar')
    _stub_days_old_cds: np.timedelta64 = np.timedelta64(30, 'D')

    def __init__(self, holidays: npt.NDArray[np.datetime64], weekmask: str | npt.NDArray[np.bool_] | None = None):
        if weekmask is None:
            self._calendar = np.busdaycalendar(holidays=holidays)
        else:
            self._calendar = np.busdaycalendar(holidays=holidays, weekmask=weekmask)

    @property
    def numpy_calendar(self) -> np.busdaycalendar:
        return self._calendar

    def _get_cds_date_range(
        self, date: np.datetime64, convention: Convention, initial_date: bool
    ) -> npt.NDArray[np.datetime64]:
        if initial_date:
            previous_20_date = previous_twentieth(date, Rule.CDS_2015)
            if self.offset(previous_20_date, 0, convention) > date:
                return previous_20_date.astype('datetime64[M]') - np.array([0, 3], dtype='timedelta64[M]')

            else:
                return previous_20_date.astype('datetime64[M]') + np.array([0, 3], dtype='timedelta64[M]')

        else:
            next_20_date = next_twentieth(date, Rule.CDS_2015)
            if self.offset(next_20_date, 0, convention) < date:
                return next_20_date.astype('datetime64[M]') + np.array([0, 3], dtype='timedelta64[M]')

            else:
                return next_20_date.astype('datetime64[M]') - np.array([3, 0], dtype='timedelta64[M]')

    def _daily_cds_2015(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64 | npt.NDArray[np.timedelta64],
        convention: Convention,
        termination_convention: Convention,
    ) -> npt.NDArray[np.datetime64]:
        first_dates = self._get_cds_date_range(effective_date, convention, True)
        final_dates = self._get_cds_date_range(termination_date, termination_convention, False)
        dates: npt.NDArray[np.datetime64] = np.arange(
            first_dates[1], final_dates[0] + period, period, dtype='datetime64[D]'
        )
        return np.r_[first_dates[0], dates, final_dates[-1]]

    def _daily_old_cds(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64 | npt.NDArray[np.timedelta64],
        termination_convention: Convention,
        *_,
    ) -> npt.NDArray[np.datetime64]:
        final_dates = self._get_cds_date_range(termination_date, termination_convention, False)
        next_twentieth_date = next_twentieth(effective_date, Rule.old_CDS)
        if next_twentieth_date - effective_date < np.timedelta64(30, 'D'):
            next_twentieth_date = next_twentieth(next_twentieth_date + np.timedelta64(1, 'D'), Rule.old_CDS)
        if next_twentieth_date != effective_date:
            dates = np.arange(next_twentieth_date, final_dates[0] + period, period) + np.timedelta64(19, 'D')
            return np.r_[effective_date, dates, final_dates[-1]]
        else:
            dates = np.arange(effective_date, final_dates[0] + period, period)
            return np.r_[dates, final_dates[-1]]

    def _monthly_cds_2015(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64 | npt.NDArray[np.timedelta64],
        convention: Convention,
        termination_convention: Convention,
    ) -> npt.NDArray[np.datetime64]:
        first_dates = self._get_cds_date_range(effective_date, convention, True)
        final_dates = self._get_cds_date_range(termination_date, termination_convention, False)
        dates: npt.NDArray[np.datetime64] = np.arange(
            first_dates[1], final_dates[0] + period, period, dtype='datetime64[M]'
        )
        return np.r_[first_dates[0], dates, final_dates[-1]] + np.timedelta64(19, 'D')

    def _monthly_old_cds(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64 | npt.NDArray[np.timedelta64],
        termination_convention: Convention,
        *_,
    ) -> npt.NDArray[np.datetime64]:
        final_dates = self._get_cds_date_range(termination_date, termination_convention, False)
        next_twentieth_date = next_twentieth(effective_date, Rule.old_CDS)
        if next_twentieth_date - effective_date < np.timedelta64(30, 'D'):
            next_twentieth_date = next_twentieth(next_twentieth_date + np.timedelta64(1, 'D'), Rule.old_CDS)
        if next_twentieth_date != effective_date:
            dates = np.arange(
                next_twentieth_date.astype('datetime64[M]'), final_dates[0] + period, period
            ) + np.timedelta64(19, 'D')
            return np.r_[effective_date, dates, final_dates[-1] + np.timedelta64(19, 'D')]
        else:
            dates = np.arange(effective_date.astype('datetime64[M]'), final_dates[0] + period, period) + np.timedelta64(
                19, 'D'
            )
            dates[0] = effective_date
            return np.r_[dates, final_dates[-1] + np.timedelta64(19, 'D')]

    def _monthly_date_generation(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64,
        end_of_month: bool,
        rule: Rule = Rule.backward,
        convention: Convention = Convention.unadjusted,
        termination_convention: Convention = Convention.unadjusted,
    ) -> npt.NDArray[np.datetime64]:
        match rule:
            case Rule.forward:
                start_date = effective_date.astype('datetime64[M]')
                end_date = termination_date.astype('datetime64[M]')

                dates = np.arange(start_date, end_date, period)

                if end_of_month:
                    dates = add_month_day(dates, 31)
                    dates[0] = effective_date
                else:
                    dates = add_month_day(dates, day(effective_date))

                if dates[-1] != termination_date:
                    dates = np.r_[dates, termination_date]

            case Rule.backward:
                start_date = effective_date.astype('datetime64[M]')
                end_date = termination_date.astype('datetime64[M]')
                dates = np.arange(end_date, start_date - period, -period)

                if end_of_month:
                    dates = add_month_day(dates, 31)
                    dates[0] = termination_date
                else:
                    dates = add_month_day(dates, day(termination_date))

                if dates[-1] != effective_date:
                    dates = np.r_[dates, effective_date]

                dates = dates[::-1]

            case Rule.CDS_2015:
                dates = self._monthly_cds_2015(
                    effective_date, termination_date, period, convention, termination_convention
                )

            case Rule.CDS:
                dates = self._monthly_cds_2015(
                    effective_date, termination_date, period, convention, termination_convention
                )

            case Rule.old_CDS:
                dates = self._monthly_old_cds(
                    effective_date, termination_date, period, convention, termination_convention
                )

            case Rule.zero:
                dates = np.array([effective_date, termination_date])

            case _:
                raise ValueError('Unknown rule')

        return dates

    def _date_daily_generation(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64 | npt.NDArray[np.timedelta64],
        _: bool,
        rule: Rule,
        convention: Convention,
        termination_convention: Convention,
    ) -> npt.NDArray[np.datetime64]:
        match rule:
            case Rule.forward:
                dates = np.arange(effective_date, termination_date, period, dtype='datetime64[D]')
                if dates[-1] != termination_date:
                    dates = np.r_[dates, termination_date]

            case Rule.backward:
                dates = np.arange(termination_date, effective_date, -period, dtype='datetime64[D]')
                if dates[-1] != effective_date:
                    dates = np.r_[dates, effective_date]

                dates = dates[::-1]

            case Rule.CDS_2015:
                dates = self._daily_cds_2015(
                    effective_date, termination_date, period, convention, termination_convention
                )

            case Rule.CDS:
                dates = self._daily_cds_2015(
                    effective_date, termination_date, period, convention, termination_convention
                )

            case Rule.old_CDS:
                dates = self._daily_old_cds(
                    effective_date, termination_date, period, convention, termination_convention
                )

            case Rule.zero:
                dates = np.array([effective_date, termination_date])

            case _:
                raise NotImplementedError(f'Rule {rule} is not implemented.')

        return dates

    @overload
    def offset(
        self,
        dates: np.datetime64,
        offset: int | np.timedelta64,
        roll: Convention = Convention.unadjusted,
    ) -> np.datetime64: ...

    @overload
    def offset(
        self,
        dates: np.datetime64,
        offset: npt.NDArray[np.int_] | npt.NDArray[np.timedelta64],
        roll: Convention = Convention.unadjusted,
    ) -> npt.NDArray[np.datetime64]: ...

    @overload
    def offset(
        self,
        dates: npt.NDArray[np.datetime64],
        offset: int | np.timedelta64,
        roll: Convention = Convention.unadjusted,
    ) -> np.datetime64 | npt.NDArray[np.datetime64]: ...

    @overload
    def offset(
        self,
        dates: npt.NDArray[np.datetime64],
        offset: npt.NDArray[np.int_] | npt.NDArray[np.timedelta64],
        roll: Convention = Convention.unadjusted,
    ) -> npt.NDArray[np.datetime64]: ...

    def offset(self, dates, offset, roll: Convention = Convention.unadjusted):
        if roll == Convention.unadjusted:
            return dates + offset

        return np.busday_offset(dates + offset, 0, roll.value, busdaycal=self._calendar)

    @overload
    def working_days_offset(
        self,
        dates: np.datetime64,
        offset: np.timedelta64,
        roll: Convention = Convention.unadjusted,
    ) -> np.datetime64: ...

    @overload
    def working_days_offset(
        self,
        dates: np.datetime64,
        offset: npt.NDArray[np.timedelta64],
        roll: Convention = Convention.unadjusted,
    ) -> npt.NDArray[np.datetime64]: ...

    @overload
    def working_days_offset(
        self,
        dates: npt.NDArray[np.datetime64],
        offset: np.timedelta64 | npt.NDArray[np.timedelta64],
        roll: Convention = Convention.unadjusted,
    ) -> npt.NDArray[np.datetime64]: ...

    def working_days_offset(self, dates, offset, roll: Convention = Convention.unadjusted):
        if roll == Convention.unadjusted:
            return np.busday_offset(dates, offset)

        return np.busday_offset(dates, offset, roll.value, busdaycal=self._calendar)

    def make_schedule(
        self,
        effective_date: np.datetime64,
        termination_date: np.datetime64,
        period: np.timedelta64,
        convention: Convention,
        termination_convention: Convention,
        end_of_month: bool,
        rule: Rule = Rule.backward,
        first_date: np.datetime64 | None = None,
        next_to_last_date: np.datetime64 | None = None,
    ) -> npt.NDArray[np.datetime64]:
        start_date: np.datetime64
        end_date: np.datetime64
        if is_first_date_not_none := first_date is not None and rule != Rule.zero:
            start_date = first_date
        else:
            start_date = effective_date

        if (
            is_next_to_last_date_not_none := next_to_last_date is not None and rule != Rule.zero
        ) and start_date < next_to_last_date:
            end_date = next_to_last_date
        else:
            end_date = termination_date
        _convention = convention

        if rule == Rule.CDS_2015:
            end_of_month = False

        _end_of_month = end_of_month
        match period.dtype:
            case 'm8[D]':
                make_schedule_function = self._date_daily_generation
            case 'm8[W]':
                make_schedule_function = self._date_daily_generation
            case 'm8[M]':
                if _end_of_month and convention != Convention.unadjusted:
                    _convention = Rule.backward
                make_schedule_function = self._monthly_date_generation
            case 'm8[Y]':
                if _end_of_month and convention != Convention.unadjusted:
                    _convention = Rule.backward
                make_schedule_function = self._monthly_date_generation
            case _:
                raise NotImplemented  # (f'Period {period} of type {period.dtype} is not implemented.')

        dates = make_schedule_function(
            start_date, end_date, period, _end_of_month, rule, convention, termination_convention
        )

        if is_first_date_not_none:
            if convention == Convention.unadjusted:
                dates = np.r_[effective_date, dates]

            else:
                dates = np.r_[np.busday_offset(effective_date, 0, roll=convention, busdaycal=self._calendar), dates]

        if is_next_to_last_date_not_none:
            if convention == Convention.unadjusted:
                dates = np.r_[dates, termination_date]
            else:
                dates = np.r_[dates, np.busday_offset(termination_date, 0, roll=convention, busdaycal=self._calendar)]
        ind = 0
        if rule == Rule.old_CDS:
            ind = 1

        dates[ind:-1] = self.offset(dates[ind:-1], 0, _convention)

        if _convention != convention and rule not in [Rule.CDS_2015, Rule.old_CDS]:
            dates[0] = np.busday_offset(effective_date, 0, roll=convention, busdaycal=self._calendar)

        dates[-1] = self.offset(dates[-1], 0, termination_convention)

        return np.unique(dates)

    def until(self, dates: npt.NDArray[np.datetime64], until_date: np.datetime64) -> npt.NDArray[np.datetime64]:
        if dates.shape[0] == 0:
            raise ValueError('Dates must have at least one date')

        return np.unique(np.r_[dates[dates <= until_date], until_date])

    def after(self, dates: npt.NDArray[np.datetime64], from_date: np.datetime64) -> npt.NDArray[np.datetime64]:
        if dates.shape[0] == 0:
            raise ValueError('Dates must have at least one date')

        return np.unique(np.r_[from_date, dates[dates >= from_date]])
