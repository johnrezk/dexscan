import bz2
import time
from datetime import datetime, timedelta
from decimal import Decimal
from threading import Lock
from types import TracebackType
from typing import Any, Sequence, Type, TypeVar

import orjson
import polars as pl
from attrs import define

T = TypeVar("T")


def batched(my_list: Sequence[T], n: int) -> Sequence[Sequence[T]]:
    return [my_list[i * n : (i + 1) * n] for i in range((len(my_list) + n - 1) // n)]


def percent_change(orig: Decimal | int, new: Decimal | int) -> Decimal:
    if isinstance(orig, int):
        orig = Decimal(orig)
    if orig == 0:
        raise ValueError("orig value must not be zero")
    if new == orig:
        return Decimal(0)
    return 100 * (new - orig) / orig


@define
class RateLimiter:
    _lock: Lock
    _delta: timedelta
    _last_run: datetime | None

    @classmethod
    def create(cls, delta: timedelta):
        return RateLimiter(Lock(), delta, None)

    def __enter__(self) -> None:
        self._lock.acquire()
        if self._last_run:
            time_passed = datetime.utcnow() - self._last_run
            if time_passed < self._delta:
                dif = self._delta - time_passed
                time.sleep(dif.total_seconds())
        self._last_run = datetime.utcnow()

    def __exit__(
        self,
        et: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._lock.release()


def split_range(first: int, last: int, size: int) -> Sequence[tuple[int, int]]:
    dif = last - first
    if dif < 0:
        raise ValueError(f"invalid range: {first}-{last}")
    iter_count = dif // size
    subranges: list[tuple[int, int]] = []
    for i in range(iter_count):
        range_start = first + i * size
        subranges.append((range_start, range_start + size - 1))
    range_start = first + iter_count * size
    subranges.append((range_start, last))
    return subranges


def df_to_json(df: pl.DataFrame) -> bytes:
    return bz2.compress(orjson.dumps(df.to_dict(as_series=False)))


def json_to_df(data: bytes, schema: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        data=orjson.loads(bz2.decompress(data)),
        schema=schema,
    )
