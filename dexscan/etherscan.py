from datetime import timedelta
from typing import Iterable, Mapping

import cattrs
import orjson
import requests
from attrs import frozen
from yarl import URL

from dexscan.primitives import Hex
from dexscan.utils import RateLimiter

# INTERNAL GLOBALS

_rate_limiter = RateLimiter.create(timedelta(seconds=10))

_conv = cattrs.Converter()
_conv.register_structure_hook(Hex, lambda v, _: Hex(v))


# SHAPES


@frozen
class ContractInfo:
    contractAddress: Hex
    contractCreator: Hex
    txHash: Hex


@frozen
class ContractRes:
    status: int
    message: str
    result: tuple[ContractInfo, ...]


# FUNCS


def get_contract_infos(
    addrs: Iterable[Hex],
) -> Mapping[Hex, ContractInfo]:
    addrs_list = list(addrs)
    if len(addrs_list) > 5:
        raise RuntimeError("can submit upto 5 addresses")
    if len(addrs_list) == 0:
        raise RuntimeError("must submit at least 1 address")
    url = URL.build(
        scheme="https",
        host="api.etherscan.io",
        path="/api",
        query={
            "module": "contract",
            "action": "getcontractcreation",
            "contractaddresses": ",".join(str(addr) for addr in addrs_list),
        },
    )
    with _rate_limiter:
        res = requests.get(str(url))
        if res.status_code != 200:
            raise RuntimeError(f"bad res {res.status_code}: {res.reason}")
        bcontent = bytes(res.content)
    content = _conv.structure(orjson.loads(bcontent), ContractRes)
    return {c.contractAddress: c for c in content.result}


def get_contract_info(addr: Hex) -> ContractInfo:
    return get_contract_infos([addr])[addr]
