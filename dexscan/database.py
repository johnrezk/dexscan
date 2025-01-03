from datetime import datetime, timedelta
from decimal import Decimal
from threading import Lock
from typing import Iterable, Mapping, Sequence

import edgedb
from attrs import define
from edgedb.blocking_client import Iteration

import dexscan.edge_codegen as cg
from dexscan.constants import MIN_BLOCK_NUM
from dexscan.primitives import Hex

# SHAPES


@define
class Token:
    addr: Hex
    ticker: str
    name: str
    decimals: int
    creator_addr: Hex
    first_block_num: int
    first_tx_hash: Hex

    def raw_to_dec(self, raw_amt: int) -> Decimal:
        return Decimal(raw_amt) / Decimal(10**self.decimals)


@define
class TokenPair:
    addr: Hex
    base_token_addr: Hex
    quote_token_addr: Hex
    first_block_num: int


@define
class TokenTransfer:
    tx_hash: Hex
    token_addr: Hex
    from_addr: Hex
    to_addr: Hex
    amount: int
    block_num: int


@define
class BalanceReq:
    token_addr: Hex
    holder_addr: Hex
    block_num: int


@define
class Swap:
    pair_addr: Hex
    block_number: int
    tx_hash: Hex
    maker_addr: Hex
    base_amt_change: int
    quote_amt_change: int
    timestamp: datetime

    @property
    def price(self) -> Decimal:
        if self.base_amt_change == 0:
            return Decimal(0)
        return abs(Decimal(self.quote_amt_change) / Decimal(self.base_amt_change))


# CLIENT CLASSES


@define
class Database:
    _exec: edgedb.Executor

    def get_token(self, token_addr: Hex) -> Token | None:
        raw = cg.get_token(self._exec, token_addr=str(token_addr))
        if raw is None:
            return None
        return Token(
            addr=Hex(raw.addr),
            ticker=raw.ticker,
            name=raw.name,
            decimals=raw.decimals,
            first_block_num=raw.first_block_num,
            first_tx_hash=Hex(raw.first_tx_hash),
            creator_addr=Hex(raw.creator_addr),
        )

    def get_pair(self, pair_addr: Hex):
        raw = cg.get_pair(self._exec, pair_addr=str(pair_addr))
        if raw is None:
            return None
        return TokenPair(
            addr=Hex(raw.addr),
            base_token_addr=Hex(raw.base_token_addr),
            quote_token_addr=Hex(raw.quote_token_addr),
            first_block_num=raw.first_block_num,
        )

    def get_latest_transfer_block_num(self, token_addr: Hex) -> int:
        return cg.get_latest_transfer_block_num(
            executor=self._exec,
            token_addr=str(token_addr),
        )

    def get_nearest_block(self, dt: datetime) -> tuple[int, datetime]:
        res = cg.get_block_at_time(self._exec, dt=dt)
        return (res.number, res.timestamp)

    def get_latest_swap_block_num(self, pair_addr: Hex) -> int:
        return cg.get_latest_swap_block_num(self._exec, pair_addr=str(pair_addr))

    def get_swaps(self, pair_addr: Hex, upto_block: int) -> tuple[Swap, ...]:
        raw = cg.get_swaps(
            executor=self._exec,
            pair_addr=str(pair_addr),
            upto_block_num=upto_block,
        )
        return tuple(
            Swap(
                pair_addr=pair_addr,
                block_number=rs.block_number,
                timestamp=rs.timestamp,
                tx_hash=Hex(rs.tx_hash),
                maker_addr=Hex(rs.actor_addr),
                base_amt_change=rs.base_amt_change,
                quote_amt_change=rs.quote_amt_change,
            )
            for rs in raw
        )

    def get_block_ts(self, block_num: int) -> datetime | None:
        res = cg.get_block_ts(self._exec, block_num=block_num)
        if res is None:
            return None
        return res.timestamp

    def get_block_num_at_dt(self, dt: datetime) -> int | None:
        res = cg.get_block_at_time(self._exec, dt=dt)
        if res is None or res.timestamp is None:
            return None
        if (dt - res.timestamp) > timedelta(seconds=30):
            return None
        return res.number

    def get_latest_db_block_num(self) -> int:
        bn = cg.get_latest_db_block_num(self._exec)
        return bn or MIN_BLOCK_NUM

    def get_block_nums_missing_ts(self) -> list[int]:
        raw = cg.get_blocks_missing_ts(self._exec)
        return [b.number for b in raw]

    def get_token_balance(self, token_addr: Hex, holder_addr: Hex, upto_blocknum: int):
        return cg.get_token_balance(
            self._exec,
            target_addr=str(holder_addr),
            upto_block_num=upto_blocknum,
            token_addr=str(token_addr),
        )

    def get_holder_addrs(self, token_addr: Hex, block_num: int) -> Sequence[Hex]:
        res = cg.get_holder_addrs(
            self._exec,
            token_addr=str(token_addr),
            block_num=block_num,
        )
        return tuple(Hex(addr) for addr in res)

    def get_all_holder_balances(
        self, token_addr: Hex, at_block: int
    ) -> Mapping[Hex, int]:
        res = cg.get_all_holder_balances(
            self._exec,
            token_addr=str(token_addr),
            upto_block=at_block,
        )
        return {Hex(item.holder_addr): item.balance for item in res}

    def get_many_balances(self, reqs: Iterable[BalanceReq]) -> Mapping[Hex, int]:
        res = cg.get_many_balances(
            self._exec,
            inputs=[
                (str(req.token_addr), str(req.holder_addr), req.block_num)
                for req in reqs
            ],
        )
        return {Hex(r.holder_addr): r.balance_amt for r in res}

    def get_pair_balances(self, pair_addr: Hex, lower_bn: int, upper_bn: int):
        return cg.get_pair_balances(
            self._exec,
            pair_addr=str(pair_addr),
            lower_bn=lower_bn,
            upper_bn=upper_bn,
        )

    def insert_block(self, block_num: int, timestamp: datetime):
        cg.insert_block(
            self._exec,
            block_num=block_num,
            timestamp=timestamp,
        )

    def insert_blocks(self, blocks: Iterable[tuple[int, datetime]]):
        cg.insert_blocks(
            self._exec,
            new_blocks=list(blocks),
        )

    def get_all_transfers(
        self, token_addr: Hex, block_num: int
    ) -> tuple[TokenTransfer, ...]:
        raw_transfers = cg.get_all_transfers(
            self._exec,
            token_addr=str(token_addr),
            block_num=block_num,
        )
        return tuple(
            TokenTransfer(
                tx_hash=Hex(t.tx_hash),
                token_addr=token_addr,
                from_addr=Hex(t.from_addr),
                to_addr=Hex(t.to_addr),
                amount=t.amount,
                block_num=t.block_number,
            )
            for t in raw_transfers
        )

    def get_watched_pairs(self) -> tuple[Hex, ...]:
        return tuple(Hex(p.addr) for p in cg.get_watched_pairs(self._exec))

    def set_pair_watching(self, pair_addr: Hex, is_watching: bool):
        cg.set_pair_watching(
            self._exec,
            pair_addr=str(pair_addr),
            is_watching=is_watching,
        )


_tx_lock = Lock()


@define
class DatabaseTx(Database):
    def __enter__(self):
        assert isinstance(self._exec, Iteration)
        _tx_lock.acquire()
        self._exec.__enter__()
        return self

    def __exit__(self, extype, ex, tb):
        assert isinstance(self._exec, Iteration)
        self._exec.__exit__(extype, ex, tb)
        _tx_lock.release()

    def insert_token(
        self,
        addr: Hex,
        ticker: str,
        name: str,
        decimals: int,
        creator_addr: Hex,
        first_tx_hash: Hex,
        first_block_num: int,
    ):
        cg.insert_token(
            self._exec,
            addr=str(addr),
            ticker=ticker,
            name=name,
            decimals=decimals,
            creator_addr=str(creator_addr),
            first_tx_hash=str(first_tx_hash),
            first_block_num=first_block_num,
        )

    def insert_pair(
        self,
        pair_addr: Hex,
        base_addr: Hex,
        quote_addr: Hex,
        first_block_num: int,
    ):
        cg.insert_pair(
            self._exec,
            pair_addr=str(pair_addr),
            base_token_addr=str(base_addr),
            quote_token_addr=str(quote_addr),
            first_block_num=first_block_num,
        )

    def insert_token_transfer(
        self,
        tx_hash: Hex,
        token_addr: Hex,
        from_addr: Hex,
        to_addr: Hex,
        amount: int,
        block_num: int,
        log_index: int,
    ):
        cg.insert_token_transfer(
            self._exec,
            tx_hash=str(tx_hash),
            token_addr=str(token_addr),
            from_addr=str(from_addr),
            to_addr=str(to_addr),
            amount=amount,
            block_num=block_num,
            log_index=log_index,
        )

    def insert_swap(
        self,
        tx_hash: Hex,
        pair_addr: Hex,
        actor_addr: Hex,
        block_num: int,
        base_amt_change: int,
        quote_amt_change: int,
    ):
        cg.insert_swap(
            self._exec,
            tx_hash=str(tx_hash),
            pair_addr=str(pair_addr),
            actor_addr=str(actor_addr),
            block_num=block_num,
            base_amt_change=base_amt_change,
            quote_amt_change=quote_amt_change,
        )

    def set_transfer_sync_upto_bn(self, token_addr: Hex, block_num: int):
        cg.set_transfers_synced_upto(
            self._exec,
            token_addr=str(token_addr),
            block_num=block_num,
        )

    def set_swap_sync_upto_bn(self, pair_addr: Hex, block_num: int):
        cg.set_swaps_synced_upto(
            self._exec,
            pair_addr=str(pair_addr),
            block_num=block_num,
        )

    def reset_token_pair(self, pair_addr: Hex):
        cg.reset_pair(self._exec, pair_addr=str(pair_addr))


# SINGLETON


_client: edgedb.Client | None = None


def _get_edge_client():
    global _client
    if _client is None:
        _client = edgedb.create_client()
    return _client


def get_db() -> Database:
    return Database(_get_edge_client())


def start_tx():
    client = _get_edge_client()
    for tx in client.transaction():
        yield DatabaseTx(tx)
