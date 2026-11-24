"""
Benchmark behind the README's "Benchmark" section.

Run from the repository root: ``uv run python benchmarks/benchmark.py``

Workload: 50-year monthly schedules starting on a random day of 2000-2025, unadjusted, in a ``(schedules, 600)``
array. Each operation runs on the whole array in one call: once for a single schedule (600 dates) and once for 10,000
schedules (6,000,000 dates). Times are the best of several runs, after a warm-up call (numba compilation and the
calendar's cached tables are excluded).
"""

import platform
import subprocess
import sys
import time
from collections.abc import Callable

import numpy as np

from financialpydate.calendars import all_calendars
from financialpydate.convention import Convention
from financialpydate.date_handler import add_month_day
from financialpydate.day_counter import Actual360, ActualActual, Business252, Thirty360

YEARS = 50
PORTFOLIO = 10_000


def monthly_schedules(n_schedules: int, rng: np.random.Generator) -> np.ndarray:
    first_month = np.datetime64('2000-01') + rng.integers(0, 26 * 12, n_schedules)
    months = first_month[:, None] + np.arange(12 * YEARS)
    days = rng.integers(1, 32, n_schedules)
    out = np.empty(months.shape, 'datetime64[D]')
    for day in np.unique(days):
        out[days == day] = add_month_day(months[days == day], int(day))
    return out


def best_time(func: Callable[[], object], budget: float = 2.0) -> float:
    func()
    best, runs, started = float('inf'), 0, time.perf_counter()
    while runs < 3 or (time.perf_counter() - started < budget and runs < 1_000):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
        runs += 1
    return best


def operations(dates: np.ndarray) -> dict[str, Callable[[], object]]:
    brazil = all_calendars['Brazil']
    adjusted = brazil.offset(dates, 0, Convention.modifiedfollowing)
    start, end = adjusted[:, :-1], adjusted[:, 1:]
    return {
        'Adjust dates (modified following, Brazil)': lambda: brazil.offset(dates, 0, Convention.modifiedfollowing),
        'Payment lag (+2 business days, Brazil)': lambda: brazil.working_days_offset(adjusted, 2, Convention.following),
        'Year fraction ACT/360': lambda: Actual360()(start, end),
        'Year fraction 30/360': lambda: Thirty360()(start, end),
        'Year fraction ACT/ACT ISDA': lambda: ActualActual()(start, end),
        'Year fraction BUS/252 (Brazil)': lambda: Business252()(start, end, brazil),
    }


def calendar_load_time() -> float:
    """Milliseconds to import and build the Brazil calendar in a fresh process, after ``import financialpydate``."""
    code = (
        'import time, financialpydate; t = time.perf_counter(); '
        'from financialpydate.calendars import Brazil; Brazil.numpy_calendar; '
        'print((time.perf_counter() - t) * 1e3)'
    )
    runs = [subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True) for _ in range(5)]
    return min(float(run.stdout) for run in runs)


def main() -> None:
    rng = np.random.default_rng(0)
    one, portfolio = monthly_schedules(1, rng), monthly_schedules(PORTFOLIO, rng)
    small, large = operations(one), operations(portfolio)

    print(
        f'{platform.processor() or platform.machine()} | Python {platform.python_version()} | numpy {np.__version__}\n'
    )
    print(f'| Operation | 1 schedule ({one.size} dates) | {PORTFOLIO:,} schedules ({portfolio.size:,} dates) |')
    print('|---|---|---|')
    for name in small:
        print(f'| {name} | {best_time(small[name]) * 1e6:,.0f} µs | {best_time(large[name]) * 1e3:,.0f} ms |')
    print(
        f'\nLoad the Brazil calendar in a fresh process (after `import financialpydate`): {calendar_load_time():.0f} ms'
    )


if __name__ == '__main__':
    main()
