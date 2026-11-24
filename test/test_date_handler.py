import calendar

import numpy as np
import pytest

from financialpydate.date_handler import add_month_day


@pytest.mark.parametrize('year', [2026, 2028])
@pytest.mark.parametrize('day', [29, 30, 31])
def test_add_month_day_caps_at_month_length(year, day):
    firsts = np.arange(np.datetime64(f'{year}-01'), np.datetime64(f'{year + 1}-01')).astype('datetime64[D]')
    expected = np.array(
        [f'{year}-{m:02d}-{min(day, calendar.monthrange(year, m)[1]):02d}' for m in range(1, 13)],
        dtype='datetime64[D]',
    )
    np.testing.assert_array_equal(add_month_day(firsts, day), expected)
