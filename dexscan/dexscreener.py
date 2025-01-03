import enum
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Type

import cattrs
import orjson
import requests
from attrs import frozen
from yarl import URL
from zoneinfo import ZoneInfo

from dexscan.primitives import Hex

_TZ_EST = ZoneInfo("America/New_York")


def struct_datetime(val: Any, type: Type) -> datetime:
    if isinstance(val, int | float):
        ms_val = val / 1000
        return datetime.fromtimestamp(ms_val, tz=_TZ_EST)
    raise RuntimeError(f"unsupported dt struct type: {type}")


conv = cattrs.Converter()

conv.register_unstructure_hook(Decimal, str)

conv.register_structure_hook(datetime, struct_datetime)
conv.register_structure_hook(Decimal, lambda val, _: Decimal(val))
conv.register_structure_hook(Hex, lambda val, _: Hex(val))


@enum.unique
class TxnType(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


@frozen
class PriceChange:
    m5: Decimal
    h1: Decimal
    h6: Decimal
    h24: Decimal


@frozen
class Liquidity:
    usd: Decimal
    base: int
    quote: Decimal


@frozen
class Token:
    address: str
    name: str
    symbol: str


@frozen
class DexPair:
    chainId: str
    dexId: str
    labels: tuple[str, ...]
    baseToken: Token
    quoteToken: Token
    price: Decimal
    priceUsd: Decimal
    marketCap: int
    liquidity: Liquidity
    priceChange: PriceChange
    pairCreatedAt: datetime

    @property
    def liqRatio(self) -> Decimal:
        return round(self.marketCap / self.liquidity.usd, 1)


# DEXSCREEN WATCHER


@frozen
class DexLogAdd:
    logType: Literal["add"]
    blockNumber: int
    blockTimestamp: int
    txnHash: str
    maker: str
    logIndex: int
    amount0: Decimal
    amount1: Decimal


@frozen
class DexLogSwap:
    logType: Literal["swap"]
    blockNumber: int
    blockTimestamp: int
    txnHash: str
    maker: str
    logIndex: int
    txnType: TxnType
    priceUsd: Decimal
    volumeUsd: Decimal
    amount0: Decimal
    amount1: Decimal

    @property
    def baseValue(self) -> Decimal:
        return self.amount0

    @property
    def quoteValue(self) -> Decimal:
        return self.amount1


_fake_headers = {
    "Host": "io.dexscreener.com",
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://dexscreener.com/",
    "Origin": "https://dexscreener.com",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}


@frozen
class DexSearchPair:
    platformId: str
    dexId: str
    pairAddress: str
    labels: tuple[str, ...]
    baseToken: Token
    quoteTokenSymbol: str
    # price: Decimal
    # priceUsd: Decimal
    # h24Txns: int
    # h24VolumeUsd: Decimal
    # liquidity: Decimal
    # pairCreatedAt: int


def search_pairs(q: str):
    url = URL.build(
        scheme="https",
        host="io.dexscreener.com",
        path="/dex/search/pairs",
        query={"q": q},
    )
    res = requests.get(str(url))
    if res.status_code != 200:
        raise RuntimeError(f"search failed with status {res.status_code}: {res.reason}")
    raw_body = orjson.loads(res.content)
    assert isinstance(raw_body, dict)
    raw_pairs = raw_body.get("pairs")
    assert isinstance(raw_pairs, list)

    pairs: list[DexSearchPair] = []
    for raw_pair in raw_pairs:
        pair = conv.structure(raw_pair, DexSearchPair)
        pairs.append(pair)

    return tuple(pairs)


@frozen
class PairPoolLock:
    address: Hex
    balance: int
    endTime: int


@frozen
class PairPool:
    address: Hex
    name: str  # typically UNISWAP etc
    baseSymbol: str
    baseAddress: Hex
    totalSupply: int
    decimals: int
    locks: tuple[PairPoolLock, ...]


@frozen
class DexPairInfoTs:
    pools: tuple[PairPool, ...]


@frozen
class DexPairInfo:
    ts: DexPairInfoTs


def get_pair_info(pair_addr: Hex, base_token_addr: Hex) -> DexPairInfo:
    url = URL.build(
        scheme="https",
        host="io.dexscreener.com",
        path=f"/dex/pair-details/ethereum/{pair_addr}",
        query={
            "gp": "1",
            "ts": "1",
            "ds": "1",
            "tokenAddress": str(base_token_addr),
        },
    )
    res = requests.get(str(url))
    if res.status_code != 200:
        raise RuntimeError(
            f"get pairs failed with status {res.status_code}: {res.reason}"
        )
    return conv.structure(orjson.loads(res.content), DexPairInfo)
