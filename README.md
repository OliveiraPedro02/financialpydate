# Financial pyDate
Python library to create cashflows and derive day count conventions using python.

Dates are numpy `datetime64[D]` values and every function works on whole arrays, so a portfolio of schedules is
processed in one call instead of a Python loop.

## Quick start

```python
import numpy as np

from financialpydate.calendars import all_calendars
from financialpydate.convention import Convention
from financialpydate.day_counter import Actual360, Business252
from financialpydate.rule import Rule

brazil = all_calendars['Brazil']

schedule = brazil.make_schedule(
    effective_date=np.datetime64('2026-01-31'),
    termination_date=np.datetime64('2031-01-31'),
    period=np.timedelta64(6, 'M'),
    convention=Convention.modifiedfollowing,
    termination_convention=Convention.modifiedfollowing,
    end_of_month=True,
    rule=Rule.backward,
)
year_fractions = Business252()(schedule[:-1], schedule[1:], brazil)
accruals = Actual360()(schedule[:-1], schedule[1:])
```

Calendar names follow QuantLib, e.g. `'Target'`, `"UnitedStates['NYSE']"`, `"Brazil['Settlement']"`; `list(all_calendars)`
lists all of them. A single calendar can also be imported directly: `from financialpydate.calendars import Brazil`.

## Using it efficiently

Calendars are loaded lazily: importing `financialpydate.calendars` reads nothing, and each calendar is built the first
time it is used (1-2 ms) and then reused for the rest of the process. `all_calendars['Brazil']` and
`from financialpydate.calendars import Brazil` return the same object, so tables cached on it (such as the business-day
table behind `Business252`) are built only once.

To get the most out of that:

- **Pass arrays, not loops.** One call on 10,000 schedules is far faster than 10,000 calls on one schedule.
- **Create calendars once.** `join_calendars([...])` and `FinancialCalendar(holidays=...)` return a new object every
  time, and a new object starts with empty caches: `Business252` on a calendar joined inside a loop or a request
  rebuilds its table each time (about 18 ms instead of 0.02 ms). Build joined calendars once, at module level.
- **Expect a slower first call.** The first use of a day counter or a calendar in a process compiles numba kernels
  (cached on disk afterwards) and builds the calendar's tables, a few hundred ms in total. Later calls are fast.

### In a web service (FastAPI, Flask, ...)

Each worker process keeps its own calendars and caches, so the costs above are paid once per worker, not per request,
as long as calendars are not created inside the endpoints. Recommended setup:

```python
# calendars.py: shared by all endpoints, created once per worker
from financialpydate import join_calendars
from financialpydate.calendars import all_calendars

BRAZIL = all_calendars['Brazil']
TARGET = all_calendars['Target']
TARGET_BRAZIL = join_calendars([TARGET, BRAZIL])  # never inside an endpoint
```

```python
# main.py
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from financialpydate.day_counter import Business252

from calendars import BRAZIL, TARGET, TARGET_BRAZIL


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up before the first request: compiles the numba kernels and builds each calendar's business-day table.
    dates = np.array(['2026-01-02', '2026-02-02'], dtype='datetime64[D]')
    for calendar in (BRAZIL, TARGET, TARGET_BRAZIL):
        Business252()(dates[:1], dates[1:], calendar)
    yield


app = FastAPI(lifespan=lifespan)
```

- With gunicorn, `--preload` builds the calendars and warms up before the workers fork, so all workers share that
  memory instead of each building its own.
- Do not use `--reload` in production: it restarts the process, and so empties every cache, whenever a file changes.

## Benchmark

50-year monthly schedules (600 dates each) starting on a random day of 2000-2025, processed as one `(schedules, 600)`
array per call. Times are after the first call, so they exclude numba compilation and the calendar's cached tables.

| Operation | 1 schedule (600 dates) | 10,000 schedules (6,000,000 dates) |
|---|---|---|
| Adjust dates (modified following, Brazil) | 33 µs | 502 ms |
| Payment lag (+2 business days, Brazil) | 67 µs | 744 ms |
| Year fraction ACT/360 | 7 µs | 56 ms |
| Year fraction 30/360 | 29 µs | 288 ms |
| Year fraction ACT/ACT ISDA | 32 µs | 353 ms |
| Year fraction BUS/252 (Brazil) | 13 µs | 69 ms |

Loading the Brazil calendar in a new process takes 8 ms after `import financialpydate`.

Measured on an Intel Core i7-11800H with Python 3.14, numpy 2.4 and numba 0.64. Reproduce with
`uv run python benchmarks/benchmark.py`.
