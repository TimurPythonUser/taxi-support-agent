import hashlib
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel


class PricingError(RuntimeError):
    """The pricing service could not produce a quote."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class PriceQuote(BaseModel):
    quote_id: str
    origin: str
    destination: str
    tariff: str
    distance_km: float
    duration_min: int
    price: int
    currency: str
    valid_until: datetime
    is_estimate: bool = True


class PriceProvider(Protocol):
    async def quote(self, origin: str, destination: str, tariff: str) -> PriceQuote: ...


_TARIFF_MULTIPLIER = {"economy": 1.0, "comfort": 1.35, "business": 2.1}


def _normalize(address: str) -> str:
    return re.sub(r"[^\w]+", " ", address.lower()).strip()


class StubPriceProvider:
    """Deterministic fake of the platform pricing API.

    The same pair of addresses always gives the same price, so the demo and the evals
    are reproducible, and the agent can be checked for quoting exactly this number.
    """

    BASE_FARE = 55
    PER_KM = 14
    PER_MIN = 2
    MIN_FARE = 90

    def __init__(self, currency: str, now: Callable[[], datetime] = datetime.now) -> None:
        self._currency = currency
        self._now = now

    async def quote(self, origin: str, destination: str, tariff: str) -> PriceQuote:
        a, b = _normalize(origin), _normalize(destination)
        for address in (a, b):
            if len(address) < 3 or not re.search(r"[^\W\d_]", address):
                raise PricingError(f"address not recognized: {address!r}", retryable=False)
        if a == b:
            raise PricingError("pickup and drop-off are the same address", retryable=False)

        digest = int(hashlib.sha256(f"{a}|{b}".encode()).hexdigest(), 16)
        distance_km = round(2 + (digest % 2300) / 100, 1)  # 2–25 km
        duration_min = round(distance_km * 2.4 + 5 + (digest >> 16) % 7)
        raw = self.BASE_FARE + self.PER_KM * distance_km + self.PER_MIN * duration_min
        price = max(self.MIN_FARE, round(raw * _TARIFF_MULTIPLIER.get(tariff, 1.0) / 5) * 5)
        return PriceQuote(
            quote_id=f"Q-{uuid.uuid4().hex[:8]}",
            origin=origin,
            destination=destination,
            tariff=tariff,
            distance_km=distance_km,
            duration_min=duration_min,
            price=price,
            currency=self._currency,
            valid_until=self._now() + timedelta(minutes=10),
        )


class UnavailablePriceProvider:
    """Simulates an outage of the pricing API (used by tests and evals)."""

    async def quote(self, origin: str, destination: str, tariff: str) -> PriceQuote:
        raise PricingError("pricing service timeout", retryable=True)
