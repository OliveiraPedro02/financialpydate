"""
Regenerate ``financialpydate/calendars`` from QuantLib.

Run from the repository root with ``uv run python -m financialpydate.update_files.get_holidays``.

Each calendar is stored as a numpy weekmask plus the holidays that fall on its working weekdays. Listing every
weekend as a holiday (with ``weekmask='1111111'``) gives the same business days but makes ``np.busday_offset`` about 3x
and ``np.busday_count`` about 1.5x slower. A weekday is only marked as weekend when it is a non-business day on every
date of the range: calendars with working weekend days (China IB) or whose weekend changed (Israel, Saudi Arabia) keep
those days in the holiday list.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

CALENDARS_DIR = Path(__file__).resolve().parents[1] / 'calendars'
FIRST_DATE = np.datetime64('1901-01-01', 'D')
END_DATE = np.datetime64('2200-01-01', 'D')  # exclusive: QuantLib's last date, Date(109573), is 2199-12-31

CALENDARS = {
    "Argentina['Merval']": ('Argentina', 'Merval'),
    'Australia': ('Australia', None),
    'Brazil': ('Brazil', None),
    "Brazil['Exchange']": ('Brazil', 'Exchange'),
    "Brazil['Settlement']": ('Brazil', 'Settlement'),
    "Canada['Settlement']": ('Canada', 'Settlement'),
    "Canada['TSX']": ('Canada', 'TSX'),
    "China['IB']": ('China', 'IB'),
    "China['SSE']": ('China', 'SSE'),
    "CzechRepublic['PSE']": ('CzechRepublic', None),
    "France['Exchange']": ('France', 'Exchange'),
    "France['Settlement']": ('France', 'Settlement'),
    'Germany': ('Germany', None),
    "Germany['Eurex']": ('Germany', 'Eurex'),
    "Germany['FrankfurtStockExchange']": ('Germany', 'FrankfurtStockExchange'),
    "Germany['Settlement']": ('Germany', 'Settlement'),
    "Germany['Xetra']": ('Germany', 'Xetra'),
    "HongKong['HKEx']": ('HongKong', None),
    "Iceland['ICEX']": ('Iceland', None),
    "India['NSE']": ('India', None),
    "Indonesia['BEJ']": ('Indonesia', 'BEJ'),
    "Indonesia['JSX']": ('Indonesia', 'JSX'),
    "Israel['Settlement']": ('Israel', 'Settlement'),
    "Israel['TASE']": ('Israel', 'TASE'),
    "Italy['Exchange']": ('Italy', 'Exchange'),
    "Italy['Settlement']": ('Italy', 'Settlement'),
    'Japan': ('Japan', None),
    "Mexico['BMV']": ('Mexico', 'BMV'),
    'NullCalendar': ('NullCalendar', None),
    "Russia['MOEX']": ('Russia', 'MOEX'),
    "Russia['Settlement']": ('Russia', 'Settlement'),
    "SaudiArabia['Tadawul']": ('SaudiArabia', 'Tadawul'),
    "Singapore['SGX']": ('Singapore', 'SGX'),
    "Slovakia['BSSE']": ('Slovakia', 'BSSE'),
    "SouthKorea['KRX']": ('SouthKorea', 'KRX'),
    "SouthKorea['Settlement']": ('SouthKorea', 'Settlement'),
    'Sweden': ('Sweden', None),
    'Switzerland': ('Switzerland', None),
    "Taiwan['TSEC']": ('Taiwan', 'TSEC'),
    'Target': ('TARGET', None),
    "Ukraine['USE']": ('Ukraine', 'USE'),
    'UnitedKingdom': ('UnitedKingdom', None),
    "UnitedKingdom['Exchange']": ('UnitedKingdom', 'Exchange'),
    "UnitedKingdom['Metals']": ('UnitedKingdom', 'Metals'),
    "UnitedKingdom['Settlement']": ('UnitedKingdom', 'Settlement'),
    "UnitedStates['FederalReserve']": ('UnitedStates', 'FederalReserve'),
    "UnitedStates['GovernmentBond']": ('UnitedStates', 'GovernmentBond'),
    "UnitedStates['LiborImpact']": ('UnitedStates', 'LiborImpact'),
    "UnitedStates['NERC']": ('UnitedStates', 'NERC'),
    "UnitedStates['NYSE']": ('UnitedStates', 'NYSE'),
    "UnitedStates['Settlement']": ('UnitedStates', 'Settlement'),
    'WeekendsOnly': ('WeekendsOnly', None),
}


def quantlib_non_business_days() -> dict[str, npt.NDArray[np.datetime64]]:
    """Every non-business day (weekends included) of each calendar in [FIRST_DATE, END_DATE), from QuantLib."""
    import QuantLib as ql

    first, last = ql.Date(1, 1, 1901), ql.Date(31, 12, 2199)
    result = {}
    for key, (class_name, market) in CALENDARS.items():
        ql_class = getattr(ql, class_name)
        calendar = ql_class(getattr(ql_class, market)) if market else ql_class()
        try:
            dates = [d.ISO() for d in calendar.holidayList(first, last, includeWeekEnds=True)]
        except RuntimeError:
            dates = []
        result[key] = np.array(dates, dtype='datetime64[D]')
    return result


def split_weekends(non_business_days: npt.NDArray[np.datetime64]) -> tuple[str, npt.NDArray[np.datetime64]]:
    """
    Split a calendar's non-business days into a numpy weekmask and the remaining holidays.

    A weekday is weekend (``'0'`` in the mask) only if it is a non-business day on every date of the range, so the
    pair reproduces exactly the same business days in [FIRST_DATE, END_DATE).
    """
    days = np.arange(FIRST_DATE, END_DATE)
    off = np.isin(days, non_business_days)
    weekday = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday; 0 is Monday, as in numpy's weekmask
    always_off = np.array([off[weekday == k].all() for k in range(7)])
    weekmask = ''.join('0' if is_weekend else '1' for is_weekend in always_off)
    return weekmask, days[off & ~always_off[weekday]]


def get_file_name(key: str) -> str:
    base_name, *name_type = key.replace("']", '').split("['")
    if len(name_type) > 0:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', base_name).lower() + '_' + name_type[0].lower()

    return re.sub(r'(?<!^)(?=[A-Z])', '_', base_name).lower()


def get_class_name(key: str) -> str:
    base_name, *name_type = key.replace("']", '').split("['")
    if len(name_type) > 0:
        return base_name + '_' + name_type[0]

    return base_name


def _calendar_source(weekmask: str, holidays: npt.NDArray[np.datetime64]) -> str:
    dates = ''.join(f"'{date}'," for date in holidays.astype(str))
    return f"FinancialCalendar(holidays=np.array(({dates}), dtype='datetime64[D]'), weekmask='{weekmask}')"


def write_calendars(non_business_days: dict[str, npt.NDArray[np.datetime64]], directory: Path = CALENDARS_DIR) -> None:
    """Write one module per calendar, ``all_calendar.py`` and ``__init__.py``, then format them with ruff."""
    header = 'import numpy as np\nfrom financialpydate import FinancialCalendar\n\n\n'
    sources = {key: _calendar_source(*split_weekends(days)) for key, days in non_business_days.items()}
    imports = []
    for key, source in sources.items():
        file_name, class_name = get_file_name(key), get_class_name(key)
        imports.append(f'from financialpydate.calendars.{file_name} import {class_name}\n')
        (directory / f'{file_name}.py').write_text(f'{header}{class_name}: FinancialCalendar = {source}\n')

    entries = ''.join(f'    "{key}": {source},\n' for key, source in sources.items())
    (directory / 'all_calendar.py').write_text(f'{header}all_calendars = {{\n{entries}}}\n')
    (directory / '__init__.py').write_text(
        ''.join(imports) + 'from financialpydate.calendars.all_calendar import all_calendars\n'
    )
    subprocess.run(['ruff', 'format', '--quiet', str(directory)], check=True)


if __name__ == '__main__':
    write_calendars(quantlib_non_business_days())
