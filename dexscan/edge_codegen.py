# AUTOGENERATED FROM:
#     'queries/get_all_holder_balances.edgeql'
#     'queries/get_all_transfers.edgeql'
#     'queries/get_block_at_time.edgeql'
#     'queries/get_block_ts.edgeql'
#     'queries/get_blocks_missing_ts.edgeql'
#     'queries/get_holder_addrs.edgeql'
#     'queries/get_latest_db_block_num.edgeql'
#     'queries/get_latest_swap_block_num.edgeql'
#     'queries/get_latest_transfer_block_num.edgeql'
#     'queries/get_many_balances.edgeql'
#     'queries/get_pair.edgeql'
#     'queries/get_pair_balances.edgeql'
#     'queries/get_swaps.edgeql'
#     'queries/get_token.edgeql'
#     'queries/get_token_balance.edgeql'
#     'queries/get_watched_pairs.edgeql'
#     'queries/insert_block.edgeql'
#     'queries/insert_blocks.edgeql'
#     'queries/insert_pair.edgeql'
#     'queries/insert_swap.edgeql'
#     'queries/insert_token.edgeql'
#     'queries/insert_token_transfer.edgeql'
#     'queries/reset_pair.edgeql'
#     'queries/set_pair_watching.edgeql'
#     'queries/set_swaps_synced_upto.edgeql'
#     'queries/set_transfers_synced_upto.edgeql'
# WITH:
#     $ edgedb-py --file dexscan/edge_codegen.py --target blocking --no-skip-pydantic-validation


from __future__ import annotations
import dataclasses
import datetime
import edgedb
import uuid


StrAddr = str
StrHash = str


@dataclasses.dataclass
class GetAllHolderBalancesResult:
    id: uuid.UUID
    holder_addr: StrAddr
    balance: int


@dataclasses.dataclass
class GetAllTransfersResult:
    id: uuid.UUID
    to_addr: StrAddr
    from_addr: StrAddr
    amount: int
    tx_hash: StrHash
    block_number: int
    log_index: int
    token: GetAllTransfersResultToken


@dataclasses.dataclass
class GetAllTransfersResultToken:
    id: uuid.UUID
    addr: StrAddr
    creator_addr: StrAddr
    decimals: int
    first_tx_hash: StrHash
    name: str
    ticker: str
    first_block_number: int
    transfers_synced_to_bn: int | None


@dataclasses.dataclass
class GetBlockAtTimeResult:
    id: uuid.UUID
    number: int
    timestamp: datetime.datetime


@dataclasses.dataclass
class GetBlocksMissingTsResult:
    id: uuid.UUID
    number: int


@dataclasses.dataclass
class GetManyBalancesResult:
    id: uuid.UUID
    holder_addr: str
    balance_amt: int


@dataclasses.dataclass
class GetPairBalancesResult:
    id: uuid.UUID
    block_number: int
    base_balance: int
    quote_balance: int


@dataclasses.dataclass
class GetPairResult:
    id: uuid.UUID
    addr: StrAddr
    base_token_addr: StrAddr
    quote_token_addr: StrAddr
    first_block_num: int


@dataclasses.dataclass
class GetSwapsResult:
    id: uuid.UUID
    tx_hash: StrHash
    actor_addr: StrAddr
    block_number: int
    timestamp: datetime.datetime
    base_amt_change: int
    quote_amt_change: int


@dataclasses.dataclass
class GetTokenResult:
    id: uuid.UUID
    addr: StrAddr
    ticker: str
    name: str
    decimals: int
    creator_addr: StrAddr
    first_tx_hash: StrHash
    first_block_num: int


@dataclasses.dataclass
class GetWatchedPairsResult:
    id: uuid.UUID
    addr: StrAddr
    base_token_addr: StrAddr
    first_block_number: int


@dataclasses.dataclass
class InsertBlockResult:
    id: uuid.UUID


@dataclasses.dataclass
class InsertPairResult:
    id: uuid.UUID


@dataclasses.dataclass
class InsertSwapResult:
    id: uuid.UUID


@dataclasses.dataclass
class InsertTokenResult:
    id: uuid.UUID


@dataclasses.dataclass
class InsertTokenTransferResult:
    id: uuid.UUID


@dataclasses.dataclass
class ResetPairResult:
    id: uuid.UUID
    mod_cnt: int


def get_all_holder_balances(
    executor: edgedb.Executor,
    *,
    token_addr: str,
    upto_block: int,
) -> list[GetAllHolderBalancesResult]:
    return executor.query(
        """\
        with
          token_addr := <str> $token_addr,
          upto_block := <int32> $upto_block,
          token := assert_exists((select Token filter .addr = token_addr)),
          transfers := (
            select TokenTransfer
            filter
              .token = token 
              and .block_number <= upto_block
              and .to_addr != .from_addr
          ),
          holder_addrs := (
            (select transfers.to_addr) union (select transfers.from_addr)
          )
        for holder_addr in holder_addrs
        union (
          with
            recv_transfers := (
              select transfers filter .to_addr = holder_addr
            ),
            send_transfers := (
              select transfers filter .from_addr = holder_addr
            )
          select {
            holder_addr := holder_addr,
            balance := (
              <bigint> 0 + sum(recv_transfers.amount) - sum(send_transfers.amount)
            )
          }
        )\
        """,
        token_addr=token_addr,
        upto_block=upto_block,
    )


def get_all_transfers(
    executor: edgedb.Executor,
    *,
    token_addr: str,
    block_num: int,
) -> list[GetAllTransfersResult]:
    return executor.query(
        """\
        with
          token := assert_exists((
            select Token filter .addr = <str> $token_addr
          )),
          block_num := <int32> $block_num
        select TokenTransfer {**}
        filter
          .token = token
          and .block_number <= block_num
          and .to_addr != .from_addr
        order by .block_number asc\
        """,
        token_addr=token_addr,
        block_num=block_num,
    )


def get_block_at_time(
    executor: edgedb.Executor,
    *,
    dt: datetime.datetime,
) -> GetBlockAtTimeResult:
    return executor.query_single(
        """\
        select assert_exists((
            select Block {number, timestamp}
            filter .timestamp <= <datetime> $dt
            order by .number desc
            limit 1
        ))\
        """,
        dt=dt,
    )


def get_block_ts(
    executor: edgedb.Executor,
    *,
    block_num: int,
) -> GetBlockAtTimeResult | None:
    return executor.query_single(
        """\
        select Block {number, timestamp}
        filter .number = <int32> $block_num\
        """,
        block_num=block_num,
    )


def get_blocks_missing_ts(
    executor: edgedb.Executor,
) -> list[GetBlocksMissingTsResult]:
    return executor.query(
        """\
        select Block {number}
        filter not exists .timestamp\
        """,
    )


def get_holder_addrs(
    executor: edgedb.Executor,
    *,
    token_addr: str,
    block_num: int,
) -> list[StrAddr]:
    return executor.query(
        """\
        with
          token_addr := <str> $token_addr,
          upto_block := <int32> $block_num,
          token := assert_exists((select Token filter .addr = token_addr)),
          transfers := (
            select TokenTransfer
            filter
              .token = token 
              and .block_number <= upto_block
              and .to_addr != .from_addr
          ),
        select (
          (select transfers.to_addr) union (select transfers.from_addr)
        )\
        """,
        token_addr=token_addr,
        block_num=block_num,
    )


def get_latest_db_block_num(
    executor: edgedb.Executor,
) -> int | None:
    return executor.query_single(
        """\
        select max(Block.number);\
        """,
    )


def get_latest_swap_block_num(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
) -> int:
    return executor.query_single(
        """\
        with
          pair := assert_exists((
            select TokenPair filter .addr = <str> $pair_addr
          )),
          most_recent_swap := (
            select Swap 
            filter .pair = pair
            order by .block_number desc
            limit 1
          )
        select assert_exists(
          pair.swaps_synced_to_bn
          ?? most_recent_swap.block_number
          ?? (pair.first_block_number - 10)
        )\
        """,
        pair_addr=pair_addr,
    )


def get_latest_transfer_block_num(
    executor: edgedb.Executor,
    *,
    token_addr: str,
) -> int:
    return executor.query_single(
        """\
        with
            token := assert_exists((
                select Token filter .addr = <str> $token_addr
            )),
            most_recent_transfer := (
              select TokenTransfer
              filter TokenTransfer.token = token
              order by .block_number desc
              limit 1
            )
        select assert_exists(
            token.transfers_synced_to_bn
            ?? most_recent_transfer.block_number
            ?? (token.first_block_number - 10)
        )\
        """,
        token_addr=token_addr,
    )


def get_many_balances(
    executor: edgedb.Executor,
    *,
    inputs: list[tuple[str, str, int]],
) -> list[GetManyBalancesResult]:
    return executor.query(
        """\
        with
          inputs := <array<tuple<str, str, int32>>> $inputs
        for input in array_unpack(inputs)
        union (
          select {
            holder_addr := input.1,
            balance_amt := get_token_balance(input.0, input.1, input.2)
          }
        )\
        """,
        inputs=inputs,
    )


def get_pair(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
) -> GetPairResult | None:
    return executor.query_single(
        """\
        select TokenPair {
            id,
            addr,
            base_token_addr := .base_token.addr,
            quote_token_addr := .quote_token.addr,
            first_block_num := .first_block_number
        } filter .addr = <str> $pair_addr\
        """,
        pair_addr=pair_addr,
    )


def get_pair_balances(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
    lower_bn: int,
    upper_bn: int,
) -> list[GetPairBalancesResult]:
    return executor.query(
        """\
        with
            pair := assert_exists((select TokenPair filter .addr = <str> $pair_addr)),
            lower_bn := <int32> $lower_bn,
            upper_bn := <int32> $upper_bn,
            block_range := range_unpack(range(lower_bn, upper_bn, inc_upper := true))
        for bn in block_range
        union (
            select {
                block_number := bn,
                base_balance := get_token_balance(pair.base_token.addr, pair.addr, bn),
                quote_balance := get_token_balance(pair.quote_token.addr, pair.addr, bn)
            }
        )\
        """,
        pair_addr=pair_addr,
        lower_bn=lower_bn,
        upper_bn=upper_bn,
    )


def get_swaps(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
    upto_block_num: int,
) -> list[GetSwapsResult]:
    return executor.query(
        """\
        with
            p := assert_exists((select TokenPair filter .addr = <str> $pair_addr)),
        select Swap {
            tx_hash,
            actor_addr,
            block_number,
            timestamp := assert_exists((
                select Block.timestamp
                filter Block.number = Swap.block_number
                limit 1
            )),
            base_amt_change,
            quote_amt_change
        } filter
            .pair = p and
            .block_number <= <int32> $upto_block_num
        order by .block_number asc\
        """,
        pair_addr=pair_addr,
        upto_block_num=upto_block_num,
    )


def get_token(
    executor: edgedb.Executor,
    *,
    token_addr: str,
) -> GetTokenResult | None:
    return executor.query_single(
        """\
        select Token {
            id,
            addr,
            ticker,
            name,
            decimals,
            creator_addr,
            first_tx_hash,
            first_block_num := .first_block_number
        } filter .addr = <str> $token_addr;\
        """,
        token_addr=token_addr,
    )


def get_token_balance(
    executor: edgedb.Executor,
    *,
    token_addr: str,
    target_addr: str,
    upto_block_num: int,
) -> int:
    return executor.query_single(
        """\
        with
            token_addr := <str> $token_addr,
            holder_addr := <str> $target_addr,
            block_num := <int32> $upto_block_num
        select get_token_balance(token_addr, holder_addr, block_num)\
        """,
        token_addr=token_addr,
        target_addr=target_addr,
        upto_block_num=upto_block_num,
    )


def get_watched_pairs(
    executor: edgedb.Executor,
) -> list[GetWatchedPairsResult]:
    return executor.query(
        """\
        select TokenPair {
            addr,
            base_token_addr := .base_token.addr,
            first_block_number
        }
        filter .is_watching = True\
        """,
    )


def insert_block(
    executor: edgedb.Executor,
    *,
    block_num: int,
    timestamp: datetime.datetime,
) -> InsertBlockResult | None:
    return executor.query_single(
        """\
        insert Block {
            number := <int32> $block_num,
            timestamp := <datetime> $timestamp
        }
        unless conflict on .number\
        """,
        block_num=block_num,
        timestamp=timestamp,
    )


def insert_blocks(
    executor: edgedb.Executor,
    *,
    new_blocks: list[tuple[int, datetime.datetime]],
) -> list[InsertBlockResult]:
    return executor.query(
        """\
        with
          new_blocks := <array<tuple<int32, datetime>>> $new_blocks
        for new_block in array_unpack(new_blocks)
        union (
          insert Block {
            number := new_block.0,
            timestamp := new_block.1
          }
        )\
        """,
        new_blocks=new_blocks,
    )


def insert_pair(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
    base_token_addr: str,
    quote_token_addr: str,
    first_block_num: int,
) -> InsertPairResult:
    return executor.query_single(
        """\
        insert TokenPair {
            addr := <str> $pair_addr,
            base_token := (select Token filter .addr = <str> $base_token_addr),
            quote_token := (select Token filter .addr = <str> $quote_token_addr),
            first_block_number := <int32> $first_block_num
        }\
        """,
        pair_addr=pair_addr,
        base_token_addr=base_token_addr,
        quote_token_addr=quote_token_addr,
        first_block_num=first_block_num,
    )


def insert_swap(
    executor: edgedb.Executor,
    *,
    block_num: int,
    tx_hash: str,
    pair_addr: str,
    actor_addr: str,
    base_amt_change: int,
    quote_amt_change: int,
) -> InsertSwapResult:
    return executor.query_single(
        """\
        with
            block_num := <int32> $block_num
        insert Swap {
            tx_hash := <str> $tx_hash,
            pair := (select TokenPair filter .addr = <str> $pair_addr),
            actor_addr := <str> $actor_addr,
            base_amt_change := <bigint> $base_amt_change,
            quote_amt_change := <bigint> $quote_amt_change,
            block_number := block_num
        }\
        """,
        block_num=block_num,
        tx_hash=tx_hash,
        pair_addr=pair_addr,
        actor_addr=actor_addr,
        base_amt_change=base_amt_change,
        quote_amt_change=quote_amt_change,
    )


def insert_token(
    executor: edgedb.Executor,
    *,
    addr: str,
    ticker: str,
    name: str,
    decimals: int,
    first_tx_hash: str,
    creator_addr: str,
    first_block_num: int,
) -> InsertTokenResult:
    return executor.query_single(
        """\
        insert Token {
            addr := <str> $addr,
            ticker := <str> $ticker,
            name := <str> $name,
            decimals := <int16> $decimals,
            first_tx_hash := <str> $first_tx_hash,
            creator_addr := <str> $creator_addr,
            first_block_number := <int32> $first_block_num
        }\
        """,
        addr=addr,
        ticker=ticker,
        name=name,
        decimals=decimals,
        first_tx_hash=first_tx_hash,
        creator_addr=creator_addr,
        first_block_num=first_block_num,
    )


def insert_token_transfer(
    executor: edgedb.Executor,
    *,
    tx_hash: str,
    token_addr: str,
    from_addr: str,
    to_addr: str,
    amount: int,
    block_num: int,
    log_index: int,
) -> InsertTokenTransferResult:
    return executor.query_single(
        """\
        insert TokenTransfer {
            tx_hash := <str> $tx_hash,
            token := assert_exists((select Token filter .addr = <str> $token_addr)),
            from_addr := <str> $from_addr,
            to_addr := <str> $to_addr,
            amount := <bigint> $amount,
            block_number := <int32> $block_num,
            log_index := <int32> $log_index
        }\
        """,
        tx_hash=tx_hash,
        token_addr=token_addr,
        from_addr=from_addr,
        to_addr=to_addr,
        amount=amount,
        block_num=block_num,
        log_index=log_index,
    )


def reset_pair(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
) -> ResetPairResult:
    return executor.query_single(
        """\
        with
          pair := assert_exists((select TokenPair filter .addr = <str> $pair_addr)),
          del_transfers := (
            delete TokenTransfer
            filter .token = pair.base_token
          ),
          del_swaps := (
            delete Swap
            filter .pair = pair
          ),
          upd_pair := (
            update pair
            set { swaps_synced_to_bn := {} }
          ),
          upd_token := (
            update pair.base_token
            set { transfers_synced_to_bn := {} }
          )
        select {
          mod_cnt := count(
            del_transfers union del_swaps union upd_pair union upd_token
          )
        }\
        """,
        pair_addr=pair_addr,
    )


def set_pair_watching(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
    is_watching: bool,
) -> InsertPairResult | None:
    return executor.query_single(
        """\
        update TokenPair filter .addr = <str> $pair_addr
        set {
            is_watching := <bool> $is_watching
        }\
        """,
        pair_addr=pair_addr,
        is_watching=is_watching,
    )


def set_swaps_synced_upto(
    executor: edgedb.Executor,
    *,
    pair_addr: str,
    block_num: int,
) -> InsertPairResult | None:
    return executor.query_single(
        """\
        update TokenPair
        filter .addr = <str> $pair_addr
        set {
            swaps_synced_to_bn := max({.swaps_synced_to_bn, <int32> $block_num})
        }\
        """,
        pair_addr=pair_addr,
        block_num=block_num,
    )


def set_transfers_synced_upto(
    executor: edgedb.Executor,
    *,
    token_addr: str,
    block_num: int,
) -> InsertTokenResult | None:
    return executor.query_single(
        """\
        update Token 
        filter .addr = <str> $token_addr
        set {
            transfers_synced_to_bn := max({.transfers_synced_to_bn, <int32> $block_num})
        }\
        """,
        token_addr=token_addr,
        block_num=block_num,
    )
