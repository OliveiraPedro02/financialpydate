import numpy as np
from financial_calendar import FinancialCalendar


Israel_Settlement = FinancialCalendar(
    holidays=np.array(
        (
            '2013-02-24',
            '2013-03-25',
            '2013-03-26',
            '2013-03-31',
            '2013-04-01',
            '2013-04-15',
            '2013-04-16',
            '2013-05-14',
            '2013-05-15',
            '2013-07-16',
            '2013-09-04',
            '2013-09-05',
            '2013-09-18',
            '2013-09-19',
            '2013-09-25',
            '2013-09-26',
            '2014-03-16',
            '2014-04-14',
            '2014-04-15',
            '2014-04-20',
            '2014-04-21',
            '2014-05-05',
            '2014-05-06',
            '2014-06-03',
            '2014-06-04',
            '2014-08-05',
            '2014-09-24',
            '2014-09-25',
            '2014-10-08',
            '2014-10-09',
            '2014-10-15',
            '2014-10-16',
            '2015-03-05',
            '2015-04-09',
            '2015-04-22',
            '2015-04-23',
            '2015-05-24',
            '2015-07-26',
            '2015-09-13',
            '2015-09-14',
            '2015-09-15',
            '2015-09-22',
            '2015-09-23',
            '2015-09-27',
            '2015-09-28',
            '2015-10-04',
            '2015-10-05',
            '2016-03-24',
            '2016-04-28',
            '2016-05-11',
            '2016-05-12',
            '2016-06-12',
            '2016-08-14',
            '2016-10-02',
            '2016-10-03',
            '2016-10-04',
            '2016-10-11',
            '2016-10-12',
            '2016-10-16',
            '2016-10-17',
            '2016-10-23',
            '2016-10-24',
            '2017-03-12',
            '2017-04-10',
            '2017-04-11',
            '2017-04-16',
            '2017-04-17',
            '2017-05-01',
            '2017-05-02',
            '2017-05-30',
            '2017-05-31',
            '2017-08-01',
            '2017-09-20',
            '2017-09-21',
            '2017-10-04',
            '2017-10-05',
            '2017-10-11',
            '2017-10-12',
            '2018-03-01',
            '2018-04-05',
            '2018-04-18',
            '2018-04-19',
            '2018-05-20',
            '2018-07-22',
            '2018-09-09',
            '2018-09-10',
            '2018-09-11',
            '2018-09-18',
            '2018-09-19',
            '2018-09-23',
            '2018-09-24',
            '2018-09-30',
            '2018-10-01',
            '2019-03-21',
            '2019-04-25',
            '2019-05-08',
            '2019-05-09',
            '2019-06-09',
            '2019-08-11',
            '2019-09-29',
            '2019-09-30',
            '2019-10-01',
            '2019-10-08',
            '2019-10-09',
            '2019-10-13',
            '2019-10-14',
            '2019-10-20',
            '2019-10-21',
            '2020-03-10',
            '2020-04-08',
            '2020-04-09',
            '2020-04-14',
            '2020-04-15',
            '2020-04-28',
            '2020-04-29',
            '2020-05-28',
            '2020-07-30',
            '2020-09-20',
            '2020-09-27',
            '2020-09-28',
            '2021-03-28',
            '2021-04-14',
            '2021-04-15',
            '2021-05-17',
            '2021-07-18',
            '2021-09-07',
            '2021-09-08',
            '2021-09-15',
            '2021-09-16',
            '2021-09-20',
            '2021-09-21',
            '2021-09-27',
            '2021-09-28',
            '2022-03-17',
            '2022-05-04',
            '2022-05-05',
            '2022-06-05',
            '2022-08-07',
            '2022-09-26',
            '2022-09-27',
            '2022-10-04',
            '2022-10-05',
            '2022-10-09',
            '2022-10-10',
            '2022-10-16',
            '2022-10-17',
            '2023-03-07',
            '2023-04-06',
            '2023-04-12',
            '2023-04-25',
            '2023-04-26',
            '2023-07-27',
            '2023-09-17',
            '2023-09-24',
            '2023-09-25',
            '2024-03-24',
            '2024-04-23',
            '2024-04-29',
            '2024-05-13',
            '2024-05-14',
            '2024-06-12',
            '2024-08-13',
            '2024-10-03',
            '2024-10-16',
            '2024-10-17',
            '2024-10-23',
            '2024-10-24',
            '2025-04-13',
            '2025-04-30',
            '2025-05-01',
            '2025-06-02',
            '2025-08-03',
            '2025-09-23',
            '2025-09-24',
            '2025-10-01',
            '2025-10-02',
            '2025-10-06',
            '2025-10-07',
            '2025-10-13',
            '2025-10-14',
            '2026-03-03',
            '2026-04-02',
            '2026-04-08',
            '2026-04-21',
            '2026-04-22',
            '2026-07-23',
            '2026-09-13',
            '2026-09-20',
            '2026-09-21',
            '2027-03-23',
            '2027-04-22',
            '2027-04-28',
            '2027-05-11',
            '2027-05-12',
            '2027-08-12',
            '2027-10-03',
            '2027-10-10',
            '2027-10-11',
            '2028-03-12',
            '2028-04-11',
            '2028-04-17',
            '2028-05-01',
            '2028-05-02',
            '2028-05-31',
            '2028-08-01',
            '2028-09-21',
            '2028-10-04',
            '2028-10-05',
            '2028-10-11',
            '2028-10-12',
            '2029-03-01',
            '2029-04-18',
            '2029-04-19',
            '2029-05-20',
            '2029-07-22',
            '2029-09-10',
            '2029-09-11',
            '2029-09-18',
            '2029-09-19',
            '2029-09-23',
            '2029-09-24',
            '2029-09-30',
            '2029-10-01',
            '2030-03-19',
            '2030-04-18',
            '2030-04-24',
            '2030-05-07',
            '2030-05-08',
            '2030-08-08',
            '2030-09-29',
            '2030-10-06',
            '2030-10-07',
            '2031-03-09',
            '2031-04-08',
            '2031-04-14',
            '2031-04-28',
            '2031-04-29',
            '2031-05-28',
            '2031-07-29',
            '2031-09-18',
            '2031-10-01',
            '2031-10-02',
            '2031-10-08',
            '2031-10-09',
            '2032-02-26',
            '2032-04-14',
            '2032-04-15',
            '2032-05-16',
            '2032-07-18',
            '2032-09-06',
            '2032-09-07',
            '2032-09-14',
            '2032-09-15',
            '2032-09-19',
            '2032-09-20',
            '2032-09-26',
            '2032-09-27',
            '2033-03-15',
            '2033-04-14',
            '2033-04-20',
            '2033-05-03',
            '2033-05-04',
            '2033-08-04',
            '2033-09-25',
            '2033-10-02',
            '2033-10-03',
            '2034-03-05',
            '2034-04-04',
            '2034-04-10',
            '2034-04-24',
            '2034-04-25',
            '2034-05-24',
            '2034-07-25',
            '2034-09-14',
            '2034-09-27',
            '2034-09-28',
            '2034-10-04',
            '2034-10-05',
            '2035-03-25',
            '2035-04-24',
            '2035-04-30',
            '2035-05-14',
            '2035-05-15',
            '2035-06-13',
            '2035-08-14',
            '2035-10-04',
            '2035-10-17',
            '2035-10-18',
            '2035-10-24',
            '2035-10-25',
            '2036-03-13',
            '2036-04-30',
            '2036-05-01',
            '2036-06-01',
            '2036-08-03',
            '2036-09-22',
            '2036-09-23',
            '2036-09-30',
            '2036-10-01',
            '2036-10-05',
            '2036-10-06',
            '2036-10-12',
            '2036-10-13',
            '2037-03-01',
            '2037-03-31',
            '2037-04-06',
            '2037-04-20',
            '2037-04-21',
            '2037-05-20',
            '2037-07-21',
            '2037-09-10',
            '2037-09-23',
            '2037-09-24',
            '2037-09-30',
            '2037-10-01',
            '2038-03-21',
            '2038-04-20',
            '2038-04-26',
            '2038-05-09',
            '2038-05-10',
            '2038-06-09',
            '2038-08-10',
            '2038-09-30',
            '2038-10-13',
            '2038-10-14',
            '2038-10-20',
            '2038-10-21',
            '2039-03-10',
            '2039-04-27',
            '2039-04-28',
            '2039-05-29',
            '2039-07-31',
            '2039-09-19',
            '2039-09-20',
            '2039-09-27',
            '2039-09-28',
            '2039-10-02',
            '2039-10-03',
            '2039-10-09',
            '2039-10-10',
            '2040-02-28',
            '2040-03-29',
            '2040-04-04',
            '2040-04-17',
            '2040-04-18',
            '2040-07-19',
            '2040-09-09',
            '2040-09-16',
            '2040-09-17',
            '2041-03-17',
            '2041-04-16',
            '2041-04-22',
            '2041-05-06',
            '2041-05-07',
            '2041-06-05',
            '2041-08-06',
            '2041-09-26',
            '2041-10-09',
            '2041-10-10',
            '2041-10-16',
            '2041-10-17',
            '2042-03-06',
            '2042-04-23',
            '2042-04-24',
            '2042-05-25',
            '2042-07-27',
            '2042-09-15',
            '2042-09-16',
            '2042-09-23',
            '2042-09-24',
            '2042-09-28',
            '2042-09-29',
            '2042-10-05',
            '2042-10-06',
            '2043-03-26',
            '2043-05-13',
            '2043-05-14',
            '2043-06-14',
            '2043-08-16',
            '2043-10-05',
            '2043-10-06',
            '2043-10-13',
            '2043-10-14',
            '2043-10-18',
            '2043-10-19',
            '2043-10-25',
            '2043-10-26',
            '2044-03-13',
            '2044-04-12',
            '2044-04-18',
            '2044-05-02',
            '2044-05-03',
            '2044-06-01',
            '2044-08-02',
            '2044-09-22',
            '2044-10-05',
            '2044-10-06',
            '2044-10-12',
            '2044-10-13',
        ),
        dtype='datetime64[D]',
    )
)
