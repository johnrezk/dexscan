from datetime import datetime
from decimal import Decimal

from attrs import define, field, frozen


def _conv_to_bytes(val: str | bytes) -> bytes:
    if isinstance(val, bytes):
        return val
    if isinstance(val, str) and val.startswith("0x"):
        return bytes.fromhex(val[2:])
    raise TypeError(f"unsupported type: {type(val)}")


@define(hash=True)
class Hex:
    _val: bytes = field(converter=_conv_to_bytes)

    def __str__(self) -> str:
        return f"0x{self._val.hex()}".lower()

    def __repr__(self) -> str:
        return f'Hex("{self}")'

    def __len__(self) -> int:
        return len(self._val)

    def __int__(self) -> int:
        return int.from_bytes(self._val)

    def __bytes__(self) -> bytes:
        return self._val

    def __lt__(self, other: "Hex") -> bool:
        return str(self) < str(other)

    def force_addr(self):
        return Hex(f"{int(self):#0{42}x}")

    def force_32b(self):
        if len(self) == 32:
            return self
        return Hex(f"{int(self):#0{66}x}")


@define
class EthLog:
    transactionHash: Hex
    transactionIndex: int
    blockNumber: int
    logIndex: int
    address: Hex
    data: Hex
    topics: tuple[Hex, ...]

    def get_topic(self, index: int) -> Hex:
        if index > len(self.topics) - 1:
            raise IndexError(f"no index at {index}")
        return self.topics[index]


@define
class CryoLog:
    block_number: int
    transaction_index: int
    log_index: int
    transaction_hash: Hex
    contract_address: Hex
    topic0: Hex | None
    topic1: Hex | None
    topic2: Hex | None
    topic3: Hex | None
    data: bytes | None


@frozen
class TransferEvent:
    tx_hash: Hex
    log_index: int
    contract_addr: Hex
    src_addr: Hex
    dest_addr: Hex
    amount: int
    block_number: int

    @classmethod
    def from_log(cls, log: EthLog):
        topic1 = log.get_topic(1).force_addr()
        topic2 = log.get_topic(2).force_addr()
        data = bytes(log.data)
        assert len(data) == 32
        return TransferEvent(
            tx_hash=log.transactionHash,
            log_index=log.logIndex,
            contract_addr=log.address.force_addr(),
            src_addr=topic1,
            dest_addr=topic2,
            amount=int.from_bytes(data[0:32]),
            block_number=log.blockNumber,
        )

    @classmethod
    def from_cryo_log(cls, log: CryoLog):
        assert log.topic1
        topic1 = log.topic1.force_addr()
        assert log.topic2
        topic2 = log.topic2.force_addr()
        assert log.data and len(log.data) == 32
        return TransferEvent(
            tx_hash=log.transaction_hash,
            log_index=log.log_index,
            contract_addr=log.contract_address.force_addr(),
            src_addr=topic1,
            dest_addr=topic2,
            amount=int.from_bytes(log.data[0:32]),
            block_number=log.block_number,
        )


@frozen
class SwapEvent:
    tx_hash: Hex
    log_index: int
    contract_addr: Hex
    sender_addr: Hex
    to_addr: Hex
    amount0_in: int
    amount1_in: int
    amount0_out: int
    amount1_out: int

    @classmethod
    def from_eth_log(cls, log: EthLog):
        data = bytes(log.data)
        assert len(data) == 128
        topic1 = log.get_topic(1).force_addr()
        topic2 = log.get_topic(2).force_addr()
        return SwapEvent(
            tx_hash=log.transactionHash,
            log_index=log.logIndex,
            contract_addr=log.address,
            sender_addr=topic1,
            to_addr=topic2,
            amount0_in=int.from_bytes(data[0:32]),
            amount1_in=int.from_bytes(data[32:64]),
            amount0_out=int.from_bytes(data[64:96]),
            amount1_out=int.from_bytes(data[96:128]),
        )

    @classmethod
    def from_cryo_log(cls, log: CryoLog):
        assert log.data and len(log.data) == 128
        assert log.topic1
        topic1 = log.topic1.force_addr()
        assert log.topic2
        topic2 = log.topic2.force_addr()
        return SwapEvent(
            tx_hash=log.transaction_hash,
            log_index=log.log_index,
            contract_addr=log.contract_address,
            sender_addr=topic1,
            to_addr=topic2,
            amount0_in=int.from_bytes(log.data[0:32]),
            amount1_in=int.from_bytes(log.data[32:64]),
            amount0_out=int.from_bytes(log.data[64:96]),
            amount1_out=int.from_bytes(log.data[96:128]),
        )


@frozen
class HighLevelSwap:
    pair_addr: Hex
    block_number: int
    tx_hash: Hex
    maker_addr: Hex
    base_amt_change: int
    quote_amt_change: int

    @property
    def price(self) -> Decimal:
        if self.base_amt_change == 0:
            return Decimal(0)
        return abs(Decimal(self.quote_amt_change) / Decimal(self.base_amt_change))


def _percent_change(orig: Decimal, new: Decimal) -> Decimal:
    if new == orig or orig == 0:
        return Decimal(0)
    return 100 * (new - orig) / orig


@define
class HolderFirstBuy:
    bn: int
    ts: datetime


@define
class Holder:
    addr: Hex
    holding_amt: int
    avg_buy_price: Decimal
    quote_amt_given: int
    quote_amt_recieved: int
    unrealized_quote_amt: int
    unrealized_quote_percent: float
    realized_pnl: int
    unrealized_pnl: int
    total_pnl: int
    is_active: bool
    last_action: str | None
    is_sniper: bool
    has_sold: bool
    quote_amt_spent_sniping: int
    fbuy: HolderFirstBuy | None

    def get_unreal_pc(self):
        orig_quote_amt = self.holding_amt * self.avg_buy_price
        return round(
            _percent_change(orig_quote_amt, Decimal(self.unrealized_quote_amt)), 1
        )

    def get_total_pc(self):
        orig_quote_amt = self.holding_amt * self.avg_buy_price
        return round(_percent_change(orig_quote_amt, Decimal(self.total_pnl)), 1)
