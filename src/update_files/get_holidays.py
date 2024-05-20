import QuantLib as ql
import re

minimum_date = ql.Date(1, 1, 1901)
maximum_date = ql.Date(109573)


def get_holidays(calendar: ql.Calendar) -> tuple:
    try:
        return tuple(
            date.to_date().strftime('%Y-%m-%d')
            for date in calendar.holidayList(minimum_date, maximum_date, includeWeekEnds=False)
        )
    except RuntimeError:
        print(f'{calendar.name()} has no holidays')
        return tuple()


calendars = {
    "Argentina['Merval']": ql.Argentina(ql.Argentina.Merval),
    "Brazil['Exchange']": ql.Brazil(ql.Brazil.Exchange),
    "Brazil['Settlement']": ql.Brazil(ql.Brazil.Settlement),
    "Canada['Settlement']": ql.Canada(ql.Canada.Settlement),
    "Canada['TSX']": ql.Canada(ql.Canada.TSX),
    "China['IB']": ql.China(ql.China.IB),
    "China['SSE']": ql.China(ql.China.SSE),
    "CzechRepublic['PSE']": ql.CzechRepublic(),
    "France['Exchange']": ql.France(ql.France.Exchange),
    "France['Settlement']": ql.France(ql.France.Settlement),
    "Germany['Eurex']": ql.Germany(ql.Germany.Eurex),
    "Germany['FrankfurtStockExchange']": ql.Germany(ql.Germany.FrankfurtStockExchange),
    "Germany['Settlement']": ql.Germany(ql.Germany.Settlement),
    "Germany['Xetra']": ql.Germany(ql.Germany.Xetra),
    "HongKong['HKEx']": ql.HongKong(),
    "Iceland['ICEX']": ql.Iceland(),
    "India['NSE']": ql.India(),
    "Indonesia['BEJ']": ql.Indonesia(ql.Indonesia.BEJ),
    "Indonesia['JSX']": ql.Indonesia(ql.Indonesia.JSX),
    "Israel['Settlement']": ql.Israel(ql.Israel.Settlement),
    "Israel['TASE']": ql.Israel(ql.Israel.TASE),
    "Italy['Exchange']": ql.Italy(ql.Italy.Exchange),
    "Italy['Settlement']": ql.Italy(ql.Italy.Settlement),
    'Japan': ql.Japan(),
    "Mexico['BMV']": ql.Mexico(ql.Mexico.BMV),
    'NullCalendar': ql.NullCalendar(),
    "Russia['MOEX']": ql.Russia(ql.Russia.MOEX),
    "Russia['Settlement']": ql.Russia(ql.Russia.Settlement),
    "SaudiArabia['Tadawul']": ql.SaudiArabia(ql.SaudiArabia.Tadawul),
    "Singapore['SGX']": ql.Singapore(ql.Singapore.SGX),
    "Slovakia['BSSE']": ql.Slovakia(ql.Slovakia.BSSE),
    "SouthKorea['KRX']": ql.SouthKorea(ql.SouthKorea.KRX),
    "SouthKorea['Settlement']": ql.SouthKorea(ql.SouthKorea.Settlement),
    "Taiwan['TSEC']": ql.Taiwan(ql.Taiwan.TSEC),
    'Target': ql.TARGET(),
    "Ukraine['USE']": ql.Ukraine(ql.Ukraine.USE),
    "UnitedKingdom['Exchange']": ql.UnitedKingdom(ql.UnitedKingdom.Exchange),
    "UnitedKingdom['Metals']": ql.UnitedKingdom(ql.UnitedKingdom.Metals),
    "UnitedKingdom['Settlement']": ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
    "UnitedStates['FederalReserve']": ql.UnitedStates(ql.UnitedStates.FederalReserve),
    "UnitedStates['GovernmentBond']": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    "UnitedStates['LiborImpact']": ql.UnitedStates(ql.UnitedStates.LiborImpact),
    "UnitedStates['NERC']": ql.UnitedStates(ql.UnitedStates.NERC),
    "UnitedStates['NYSE']": ql.UnitedStates(ql.UnitedStates.NYSE),
    "UnitedStates['Settlement']": ql.UnitedStates(ql.UnitedStates.Settlement),
    'WeekendsOnly': ql.WeekendsOnly(),
}


holiday_list_string = {key: get_holidays(calendar) for key, calendar in calendars.items()}


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


if __name__ == '__main__':
    for key, value in holiday_list_string.items():
        file_name = get_file_name(key)
        class_name = get_class_name(key)
        with open('../../src/calendars/' + file_name + '.py', 'w') as file:
            file.write(
                f"""import numpy as np
from financial_calendar import FinancialCalendar

    
{class_name} = FinancialCalendar(holidays=np.array({value}, dtype="datetime64[D]"))
"""
            )

    with open('../../src/calendars/all_calendar.py', 'w') as file:
        to_write = """import numpy as np
from financial_calendar import FinancialCalendar

all_calendars = {
"""

        for key, value in holiday_list_string.items():
            to_write += f'   "{key}": FinancialCalendar(holidays=np.array({value}, dtype="datetime64[D]")),\n'

        to_write += '}'
        file.write(to_write)


# holiday_list_numpy = {key: FinancialCalendar(holidays=np.array(value, dtype='datetime64[D]'))}
