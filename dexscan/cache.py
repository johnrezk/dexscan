import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
from typing import Any, Callable, Hashable, Optional, TypeVar, cast

__all__ = ("CacheManager",)

_logger = logging.getLogger(__name__)

NO_RESULT = object()
KWD_MARK = object()


@dataclass(slots=True)
class Result:
    val: Any
    time: datetime
    lock: Lock

    @classmethod
    def factory(cls):
        return cls(NO_RESULT, datetime.now(), Lock())

    def get_value(self, action: Callable[[], Any]) -> Any:
        with self.lock:
            if self.val is NO_RESULT:
                self.val = action()
            return self.val


@dataclass(slots=True)
class SingleCache:
    ttl: timedelta
    max_size: int
    clean_index: int
    results: defaultdict[Hashable, Result]
    lock: Lock

    @classmethod
    def factory(cls, ttl: timedelta, max_size: int):
        if ttl < timedelta(seconds=1):
            raise ValueError(f"ttl too short: {ttl}")
        if max_size < 32:
            raise ValueError(f"max_size too small: {max_size}")
        clean_index = max_size // 4
        return cls(ttl, max_size, clean_index, defaultdict(Result.factory), Lock())

    def invalidate_all(self):
        with self.lock:
            self.results.clear()

    def invalidate_one(self, key: Hashable):
        with self.lock:
            self.results.pop(key, None)

    def remove_all_expired(self) -> None:
        with self.lock:
            now = datetime.now()
            for key, res in tuple(self.results.items()):
                if now - res.time > self.ttl:
                    self.results.pop(key, None)

    def get_result(self, key: Hashable) -> Result:
        with self.lock:
            res = self.results[key]

            # remove result if it exceeds ttl
            if datetime.now() - res.time > self.ttl:
                self.results.pop(key, None)

            # clean cache if it exceeds max size
            if len(self.results) > self.max_size:
                items = [(k, v) for k, v in self.results.items()]
                items.sort(key=lambda x: x[1].time)
                for key, _ in items[: self.clean_index]:
                    self.results.pop(key, None)

            return self.results[key]


_F = TypeVar("_F", bound=Callable[..., Any])


def _get_result_method_key(args: tuple, kwargs: dict) -> Hashable:
    # this removes the "self" arg from args
    return args[1:] + (KWD_MARK,) + tuple(sorted(kwargs.items()))


def _get_result_function_key(args: tuple, kwargs: dict) -> Hashable:
    return args + (KWD_MARK,) + tuple(sorted(kwargs.items()))


_caches: list["CacheManager"] = []
_caches_lock = Lock()


def clean_all_caches() -> None:
    _logger.info("cleaning all caches...")
    with _caches_lock:
        for cm in _caches:
            cm._clean()
    _logger.info("done")


@dataclass(slots=True, frozen=True)
class CacheManager:
    _ttl: timedelta
    _max_size: int
    _caches: dict[Hashable, SingleCache]

    @classmethod
    def factory(cls, ttl: timedelta, max_size: int):
        with _caches_lock:
            inst = cls(ttl, max_size, {})
            _caches.append(inst)
            return inst

    def _get_method_cache(self, key: Hashable):
        cache = self._caches.get(key)
        if cache is None:
            raise RuntimeError(f"cache does not exist: {key}")
        return cache

    def invalidate_all(self, key: Hashable):
        cache = self._get_method_cache(key)
        cache.invalidate_all()

    def invalidate_one(self, key: Hashable, *args, **kwargs):
        cache = self._get_method_cache(key)
        padded_args = (None,) + args
        res_key = _get_result_method_key(padded_args, kwargs)
        cache.invalidate_one(res_key)

    def _clean(self) -> None:
        for cache in self._caches.values():
            cache.remove_all_expired()

    def method(
        self,
        key: Hashable,
        ttl: Optional[timedelta] = None,
        max: Optional[int] = None,
    ):
        m_ttl = self._ttl if ttl is None else ttl
        m_max_size = self._max_size if max is None else max

        def deco(func: _F) -> _F:
            if key in self._caches:
                raise RuntimeError(f"cache already exists with key: {key}")
            cache = SingleCache.factory(m_ttl, m_max_size)
            self._caches[key] = cache

            @wraps(func)
            def wrapped(*args, **kwargs):
                key = _get_result_method_key(args, kwargs)
                result = cache.get_result(key)

                def partial():
                    return func(*args, **kwargs)

                return result.get_value(partial)

            return cast(_F, wrapped)

        return deco

    def function(
        self,
        key: Hashable,
        ttl: timedelta | None = None,
        max: int | None = None,
    ):
        m_ttl = self._ttl if ttl is None else ttl
        m_max_size = self._max_size if max is None else max

        def deco(func: _F) -> _F:
            if key in self._caches:
                raise RuntimeError(f"cache already exists with key: {key}")
            cache = SingleCache.factory(m_ttl, m_max_size)
            self._caches[key] = cache

            @wraps(func)
            def wrapped(*args, **kwargs):
                key = _get_result_function_key(args, kwargs)
                result = cache.get_result(key)

                def partial():
                    return func(*args, **kwargs)

                return result.get_value(partial)

            return cast(_F, wrapped)

        return deco
