from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel


class Trip(BaseModel):
    order_id: str
    date: date
    time: str  # HH:MM, local time
    origin: str
    destination: str
    driver_name: str
    car: str
    price: int
    currency: str


class OrderRepository(Protocol):
    async def get(self, order_id: str) -> Trip | None: ...

    async def search(self, trip_date: date, approx_time: str, route: str | None) -> list[Trip]: ...


# (days ago, HH:MM, order id, origin, destination, driver, car, price)
# fmt: off
_FIXTURES = [
    (0, None,    "732017", "ул. Антоновича, 50",    "ТРЦ Ocean Plaza",        "Андрей", "Toyota Camry, серый",      165),
    (1, "08:15", "731655", "просп. Оболонский, 15", "Аэропорт Борисполь",     "Сергей", "Kia Optima, чёрный",       610),
    (1, "19:40", "731842", "ул. Крещатик, 22",      "Центральный ж/д вокзал", "Олег",   "Skoda Octavia, белый",     245),
    (1, "20:10", "731870", "Пейзажная аллея",       "ул. Лесковая, 9",        "Ирина",  "Hyundai Elantra, синий",   180),
    (2, "22:30", "729904", "ЖК Комфорт Таун",       "ул. Саксаганского, 110", "Максим", "Renault Megane, серый",    210),
    (3, "13:05", "728410", "ТРЦ Lavina Mall",       "просп. Победы, 37",      "Виктор", "Volkswagen Passat, синий", 275),
]
# fmt: on


class StubOrderRepository:
    """Fake order history. Dates are relative to "now", so "yesterday evening" always works in the demo."""

    SEARCH_WINDOW = timedelta(minutes=75)

    def __init__(self, tz: ZoneInfo, currency: str, now: datetime | None = None) -> None:
        now = now or datetime.now(tz)
        self._trips: dict[str, Trip] = {}
        for days_ago, hhmm, order_id, origin, destination, driver, car, price in _FIXTURES:
            moment = now - timedelta(hours=2) if hhmm is None else None
            trip_date = moment.date() if moment else (now - timedelta(days=days_ago)).date()
            trip_time = moment.strftime("%H:%M") if moment else hhmm
            self._trips[order_id] = Trip(
                order_id=order_id,
                date=trip_date,
                time=trip_time,
                origin=origin,
                destination=destination,
                driver_name=driver,
                car=car,
                price=price,
                currency=currency,
            )

    async def get(self, order_id: str) -> Trip | None:
        return self._trips.get(order_id)

    async def search(self, trip_date: date, approx_time: str, route: str | None) -> list[Trip]:
        target = datetime.combine(trip_date, time.fromisoformat(approx_time))
        matches = [
            t
            for t in self._trips.values()
            if abs(datetime.combine(t.date, time.fromisoformat(t.time)) - target) <= self.SEARCH_WINDOW
        ]
        if len(matches) > 1 and route:
            words = {w for w in route.lower().replace(",", " ").split() if len(w) > 3}
            narrowed = [t for t in matches if any(w[:5] in f"{t.origin} {t.destination}".lower() for w in words)]
            matches = narrowed or matches
        return sorted(matches, key=lambda t: (t.date, t.time))

    def all(self) -> list[Trip]:
        return list(self._trips.values())
