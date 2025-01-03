import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cattrs
import cryo
import orjson
import requests
from attrs import define
from cattrs.gen import make_dict_structure_fn, override
from web3 import Web3
from websockets.sync.client import connect as connect_websocket

from dexscan.primitives import CryoLog, EthLog, Hex
from dexscan.utils import split_range

# Private Constants


_ERC20_ABI = [
    {
        "name": "name",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
        "constant": True,
        "payable": False,
        "type": "function",
    },
    {
        "name": "decimals",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
        "constant": True,
        "payable": False,
        "type": "function",
    },
    {
        "name": "balanceOf",
        "inputs": [{"name": "_owner", "type": "address"}],
        "outputs": [{"name": "balance", "type": "uint256"}],
        "constant": True,
        "payable": False,
        "type": "function",
    },
    {
        "name": "symbol",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
        "constant": True,
        "payable": False,
        "type": "function",
    },
]

_USV2_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function",
    },
]


# Shapes


@define
class EthTx:
    hash: Hex
    blockHash: Hex
    blockNumber: int


@define
class Block:
    number: int
    hash: Hex
    timestamp: int


@define
class FinalizedBlock:
    number: int
    hash: Hex


@define
class EthTxReciept:
    transactionHash: Hex
    transactionIndex: int
    blockNumber: int
    from_addr: Hex
    to_addr: Hex | None
    contractAddress: Hex | None
    logs: tuple[EthLog, ...]


@define
class MempoolTx:
    hash: Hex
    nonce: int
    fromAddr: Hex
    toAddr: Hex | None
    input: Hex
    type: int
    # gas: int
    # gasPrice: int
    # maxFeePerGas: int
    # maxPriorityFeePerGas: int


@define
class CreatePairTx:
    tx_hash: Hex
    block_num: int
    block_ts: int
    pair_addr: Hex
    dev_addr: Hex


@define
class Erc20Info:
    name: str
    symbol: str
    decimals: int


@define
class UniV2PairInfo:
    base_token_addr: Hex
    quote_token_addr: Hex


# CONVERTER


def _unstruct_int(val) -> bool | str:
    if isinstance(val, bool):
        return val
    return hex(val)


def _struct_int(val, _) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.startswith("0x"):
        return int(val, 16)
    raise TypeError(f"unsupported type: {type(val)}")


_conv = cattrs.Converter()

# used for params sent to rpc
_conv.register_unstructure_hook(Hex, str)
_conv.register_unstructure_hook(int, _unstruct_int)

# used for results from rpc
_conv.register_structure_hook(int, _struct_int)
_conv.register_structure_hook(Hex, lambda v, _: Hex(v))
_conv.register_structure_hook(
    EthTxReciept,
    make_dict_structure_fn(
        cl=EthTxReciept,
        converter=_conv,
        from_addr=override(rename="from"),
        to_addr=override(rename="to"),
    ),
)
_conv.register_structure_hook(
    MempoolTx,
    make_dict_structure_fn(
        cl=MempoolTx,
        converter=_conv,
        fromAddr=override(rename="from"),
        toAddr=override(rename="to"),
    ),
)


_RPC_URL = "http://127.0.0.1:8545"
_WS_URL = "ws://127.0.0.1:8546"
_MAX_CONCURRENT_REQ = 16


def range_inclusive(start: int, stop: int):
    return range(start, stop + 1)


@define
class EthRPC:
    _sesh: requests.Session
    _call_semaphore: threading.Semaphore

    def _call(self, method: str, params: list[Any] | None = None) -> Any | None:
        payload = orjson.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": _conv.unstructure(params) if params else [],
                "id": 1,
            }
        )
        with self._call_semaphore:
            res = self._sesh.post(url=_RPC_URL, data=payload, timeout=10)
        if res.status_code == 429:
            raise RuntimeError("too many node connections")
        body = orjson.loads(res.content)
        assert isinstance(body, dict)
        return body["result"]

    def _collect_logs(
        self,
        block_range: tuple[int, int],
        topic0: Hex,
        topic1: Hex | None = None,
        topic2: Hex | None = None,
        contract_addr: Hex | None = None,
    ):
        res = cryo.collect(
            datatype="logs",
            output_format="list",
            rpc=_RPC_URL,
            max_concurrent_requests=_MAX_CONCURRENT_REQ,
            blocks=tuple(str(bn) for bn in range(block_range[0], block_range[1] + 1)),
            contract=str(contract_addr) if contract_addr else None,
            topic0=str(topic0),
            topic1=str(topic1) if topic1 else None,
            topic2=str(topic2) if topic2 else None,
        )
        assert isinstance(res, list)
        return _conv.structure(res, tuple[CryoLog, ...])

    def _get_web3_client(self):
        return Web3(Web3.HTTPProvider(_RPC_URL))

    def get_tx(self, tx_hash: Hex) -> EthTx | None:
        raw = self._call(
            method="eth_getTransactionByHash",
            params=[tx_hash],
        )
        if raw is None:
            return None
        return _conv.structure(raw, EthTx)

    def get_block(self, block_num: int) -> Block | None:
        raw = self._call(
            method="eth_getBlockByNumber",
            params=[str(hex(block_num)), False],
        )
        if raw is None:
            return None
        assert isinstance(raw, dict)
        return _conv.structure(raw, Block)

    def iter_blocks(self, block_range: tuple[int, int]):
        for first_bn, last_bn in split_range(block_range[0], block_range[1], 200):
            res = cryo.collect(
                datatype="blocks",
                output_format="list",
                rpc=_RPC_URL,
                max_concurrent_requests=_MAX_CONCURRENT_REQ,
                blocks=[str(bn) for bn in range_inclusive(first_bn, last_bn)],
            )
            assert isinstance(res, list)
            yield _conv.structure(res, tuple[Block, ...])

    def get_latest_block_num(self) -> int:
        raw = self._call(
            method="eth_getBlockByNumber",
            params=["latest", False],
        )
        assert isinstance(raw, dict)
        return _conv.structure(raw["number"], int)

    def iter_logs(
        self,
        block_range: tuple[int, int],
        contract_addr: Hex | None = None,
        topics: list[Hex | None | list[Hex | None]] | None = None,
    ):
        start_block, end_block = block_range
        dif = end_block - start_block
        if dif < 0:
            raise ValueError(f"invalid block range: {start_block}-{end_block}")

        def call_struct(params: dict) -> list[EthLog]:
            raw = self._call(
                method="eth_getLogs",
                params=[params],
            )
            assert isinstance(raw, list)
            if len(raw) == 0:
                return []
            return _conv.structure(raw, list[EthLog])

        all_params: list[dict] = []
        for bn in range(start_block, end_block + 1):
            params: dict[str, Any] = {"fromBlock": bn, "toBlock": bn}
            if contract_addr:
                params["address"] = contract_addr
            if topics:
                params["topics"] = topics
            all_params.append(params)

        with ThreadPoolExecutor(max_workers=24) as tp:
            for log_chunk in tp.map(call_struct, all_params):
                if len(log_chunk) == 0:
                    continue
                yield log_chunk

    def iter_logs_cryo(
        self,
        block_range: tuple[int, int],
        topic0: Hex,
        topic1: Hex | None = None,
        topic2: Hex | None = None,
        contract_addr: Hex | None = None,
    ):
        for start, end in split_range(block_range[0], block_range[1], 200):
            yield self._collect_logs(
                block_range=(start, end),
                topic0=topic0,
                topic1=topic1,
                topic2=topic2,
                contract_addr=contract_addr,
            )

    def get_tx_reciept(self, tx_hash: Hex):
        raw = self._call(
            method="eth_getTransactionReceipt",
            params=[tx_hash],
        )
        if raw is None:
            return None
        return _conv.structure(raw, EthTxReciept)

    def _get_erc20_contract(self, token_addr: Hex):
        client = self._get_web3_client()
        cs_addr = Web3.to_checksum_address(str(token_addr))
        return client.eth.contract(address=cs_addr, abi=_ERC20_ABI)

    def _get_univ2_contract(self, pair_addr: Hex):
        client = self._get_web3_client()
        cs_addr = Web3.to_checksum_address(str(pair_addr))
        return client.eth.contract(address=cs_addr, abi=_USV2_PAIR_ABI)

    def get_erc20_info(self, token_addr: Hex) -> Erc20Info:
        contract = self._get_erc20_contract(token_addr)
        decimals = contract.functions.decimals().call()
        assert isinstance(decimals, int)
        name = contract.functions.name().call()
        assert isinstance(name, str)
        symbol = contract.functions.symbol().call()
        assert isinstance(symbol, str)
        return Erc20Info(
            name=name,
            symbol=symbol,
            decimals=decimals,
        )

    def get_erc20_balance(self, token_addr: Hex, holder_addr: Hex) -> int:
        contract = self._get_erc20_contract(token_addr)
        holder_addr_cs = Web3.to_checksum_address(str(holder_addr))
        balance = contract.functions.balanceOf(holder_addr_cs).call()
        assert isinstance(balance, int)
        return balance

    def get_univ2_pair_info(self, pair_addr: Hex) -> UniV2PairInfo:
        contract = self._get_univ2_contract(pair_addr)
        btoken = contract.functions.token0().call()
        assert isinstance(btoken, str)
        qtoken = contract.functions.token1().call()
        assert isinstance(qtoken, str)
        return UniV2PairInfo(
            base_token_addr=Hex(btoken).force_addr(),
            quote_token_addr=Hex(qtoken).force_addr(),
        )

    def subscribe_mempool_txs(self):
        with connect_websocket(_WS_URL) as ws:
            ws.send(
                orjson.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newPendingTransactions", True],
                    }
                ),
            )
            # ignore first response
            ws.recv()
            while True:
                raw_res = ws.recv()
                data = orjson.loads(raw_res)
                yield _conv.structure(data["params"]["result"], MempoolTx)

    def get_created_pairs_transfer_logs(self, dev_addr: Hex | None):
        transfer_sig = Hex(
            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        )
        null_addr = Hex("0x0000000000000000000000000000000000000000")

        filtered_logs: list[CryoLog] = []
        for logs in self.iter_logs_cryo(
            block_range=(17542000, self.get_latest_block_num()),
            topic0=transfer_sig,
            topic1=null_addr.force_32b(),
            topic2=dev_addr.force_32b() if dev_addr else None,
        ):
            for log in logs:
                filtered_logs.append(log)

        pairs: list[CreatePairTx] = []
        for log in filtered_logs:
            tx = self.get_tx_reciept(log.transaction_hash)
            if tx is None:
                continue
            block = self.get_block(tx.blockNumber)
            if block is None:
                continue
            pairs.append(
                CreatePairTx(
                    tx_hash=tx.transactionHash,
                    block_num=tx.blockNumber,
                    block_ts=block.timestamp,
                    dev_addr=tx.from_addr,
                    pair_addr=Hex(""),
                )
            )


_rpc: EthRPC | None = None
_lock = threading.Lock()


def get_rpc() -> EthRPC:
    global _rpc
    with _lock:
        if _rpc is None:
            sesh = requests.Session()
            sesh.headers = {"Content-Type": "application/json"}
            _rpc = EthRPC(sesh, threading.Semaphore(24))
        return _rpc
