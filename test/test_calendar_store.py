import threading
import types

import numpy as np
import pytest

from financialpydate.calendar_store import CalendarStore
from financialpydate.calendars import all_calendars
from financialpydate.update_files.get_holidays import (
    END_DATE,
    FIRST_DATE,
    _year_ranges,
    get_class_name,
    write_calendars,
)

NAMES = ['Brazil', "China['IB']", "SaudiArabia['Tadawul']", 'NullCalendar']
DAYS = np.arange(FIRST_DATE, END_DATE)


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    """Run the generator into a temporary folder, from the calendars currently in the package."""
    directory = tmp_path_factory.mktemp('calendars')
    for stale in ('brazil.py', 'all_calendar.py', 'keep_me.py'):  # brazil and all_calendar were written by old versions
        (directory / stale).write_text('')
    write_calendars(
        {name: DAYS[~np.is_busday(DAYS, busdaycal=all_calendars[name].numpy_calendar)] for name in NAMES}, directory
    )
    return directory


def test_store_reproduces_calendars(generated):
    store = CalendarStore(generated / 'holidays.npz')
    for name in NAMES:
        np.testing.assert_array_equal(
            np.is_busday(DAYS, busdaycal=store[name].numpy_calendar),
            np.is_busday(DAYS, busdaycal=all_calendars[name].numpy_calendar),
        )


def test_store_is_lazy_and_returns_one_object(generated):
    store = CalendarStore(generated / 'holidays.npz')
    assert len(store) == len(NAMES) and list(store) == NAMES and 'Brazil' in store
    assert repr(store) == f'CalendarStore({len(NAMES)} calendars, 0 loaded)'
    assert store['Brazil'] is store['Brazil']
    assert repr(store) == f'CalendarStore({len(NAMES)} calendars, 1 loaded)'
    with pytest.raises(KeyError):
        store['Atlantis']


def test_store_builds_each_calendar_once_across_threads(generated):
    store = CalendarStore(generated / 'holidays.npz')
    results = []
    threads = [threading.Thread(target=lambda: results.append(store['Brazil'])) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({id(calendar) for calendar in results}) == 1


def test_generator_writes_only_init_and_data(generated):
    assert sorted(path.name for path in generated.iterdir()) == ['__init__.py', 'holidays.npz', 'keep_me.py']


def test_generated_init(generated):
    source = (generated / '__init__.py').read_text()
    assert "'ChinaIB': \"China['IB']\"," in source
    assert '    ChinaIB: FinancialCalendar\n' in source  # TYPE_CHECKING declaration

    module = types.ModuleType('generated_calendars')
    exec(compile(source, str(generated / '__init__.py'), 'exec'), module.__dict__)
    store = CalendarStore(generated / 'holidays.npz')
    module.all_calendars = store  # the generated module reads the package's store; point it at the generated data

    assert module.ChinaIB is store["China['IB']"]
    assert module.__all__ == ['all_calendars', 'Brazil', 'ChinaIB', 'SaudiArabiaTadawul', 'NullCalendar']
    assert 'Brazil' in dir(module)
    with pytest.raises(AttributeError, match='Atlantis'):
        module.Atlantis  # noqa: B018


@pytest.mark.skipif(not isinstance(all_calendars, CalendarStore), reason='calendars not regenerated with the store yet')
def test_package_calendars_are_shared():
    import financialpydate.calendars as calendars
    from financialpydate.calendars import Brazil

    assert all_calendars['Brazil'] is Brazil is calendars.Brazil


def test_year_ranges():
    assert _year_ranges([1901]) == '1901'
    assert _year_ranges([1901, 1902, 1903, 2026, 2028, 2029]) == '1901-1903, 2026, 2028-2029'


def test_class_names():
    assert get_class_name('Brazil') == 'Brazil'
    assert get_class_name("UnitedStates['NYSE']") == 'UnitedStatesNYSE'
    assert get_class_name("Germany['FrankfurtStockExchange']") == 'GermanyFrankfurtStockExchange'
