"""
Lazy, lock-free access to the calendars generated in ``calendars/holidays.npz``.

``financialpydate/update_files/get_holidays.py`` writes, for every calendar, its numpy weekmask and its holidays as
int32 day numbers (days since 1970-01-01) into one compressed ``.npz``. Nothing is read at import time: a calendar is
built the first time it is requested and the same `FinancialCalendar` object is returned afterwards, so its cached
tables (e.g. `FinancialCalendar.cumulative_business_days`, used by `Business252`) are built once per process.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

from financialpydate.financial_calendar import FinancialCalendar

DATA_FILE = Path(__file__).resolve().parent / 'calendars' / 'holidays.npz'


class CalendarStore(Mapping[str, FinancialCalendar]):
    """
    Read-only mapping from calendar name (e.g. ``"Brazil['Settlement']"``) to `FinancialCalendar`.

    ``store[name]`` builds the calendar on first access (1-2 ms) and returns the same object on every later call, from
    any thread. There is no lock: the data never changes, so a lookup is a plain dict read, and a new calendar is
    published with ``dict.setdefault``, which is atomic. If two threads build the same calendar at the same time, both
    get the object that was stored first. Iterating, ``len`` and ``in`` only read the list of names; ``values()`` and
    ``items()`` build every calendar.
    """

    __slots__ = ('_path', '_calendars', '_members')

    def __init__(self, path: Path = DATA_FILE):
        self._path = path
        self._calendars: dict[str, FinancialCalendar] = {}
        self._members: dict[str, str] | None = None

    def __getitem__(self, name: str) -> FinancialCalendar:
        calendar = self._calendars.get(name)
        if calendar is not None:
            return calendar
        member = self._names()[name]
        with np.load(self._path) as data:
            holidays = data[f'{member}_holidays'].astype('datetime64[D]')
            weekmask = data[f'{member}_weekmask']
        return self._calendars.setdefault(name, FinancialCalendar(holidays=holidays, weekmask=weekmask))

    def __iter__(self) -> Iterator[str]:
        return iter(self._names())

    def __len__(self) -> int:
        return len(self._names())

    def __contains__(self, name: object) -> bool:
        return name in self._names()

    def __repr__(self) -> str:
        return f'{type(self).__name__}({len(self)} calendars, {len(self._calendars)} loaded)'

    def _names(self) -> dict[str, str]:
        """Calendar name -> member prefix in the ``.npz``, read once (a race only reads the same names twice)."""
        members = self._members
        if members is None:
            with np.load(self._path) as data:
                members = dict(zip(data['names'].tolist(), data['members'].tolist(), strict=True))
            self._members = members
        return members


all_calendars = CalendarStore()
