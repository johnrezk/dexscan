import enum
import math
import statistics
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Hashable, Iterable, Literal, Mapping, Sequence

import web3
from attrs import define, frozen
from zoneinfo import ZoneInfo

from dexscan.cache import CacheManager
from dexscan.constants import MIN_BLOCK_NUM
from dexscan.database import Swap, Token, TokenPair, TokenTransfer, get_db, start_tx
from dexscan.etherscan import get_contract_info
from dexscan.primitives import (
    Hex,
    HighLevelSwap,
    Holder,
    HolderFirstBuy,
    SwapEvent,
    TransferEvent,
)
from dexscan.rpc import get_rpc
from dexscan.uniswap import UniswapV2Calculator
from dexscan.utils import percent_change, split_range

# CONSTANTS

TZ_EST = ZoneInfo("America/New_York")
WETH_ADDR = Hex("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
TRANSFER_SIG = Hex("0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")
BLACKLISTED_ADDRS: set[Hex] = {
    Hex("0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13"),  # jaredfromsubway.eth
    Hex("0x0000000000000000000000000000000000000000"),  # null address
    Hex("0x000000000000000000000000000000000000dEaD"),  # dead address
}
MKTCAP_ETH_AMT = round(0.08 * (10**18))

# PRIVATE UTILS


def _get_event_sig(val: str) -> Hex:
    return Hex(bytes(web3.Web3.keccak(text=val)))


# SHAPES


@frozen
class PairState:
    block_num: int
    block_dt: datetime
    quote_token: Token
    swaps: tuple[Swap, ...]
    holders: tuple[Holder, ...]
    price_calc: UniswapV2Calculator
    first_swap_block_num: int
    first_swap_block_dt: datetime


@define
class ProfitLossSpread:
    p500: float
    p400: float
    p300: float
    p250: float
    p200: float
    p150: float
    p100: float
    p80: float
    p60: float
    p40: float
    p20: float
    breakeven: float
    l20: float
    l40: float
    l60: float
    l80: float

    @classmethod
    def create_empty(cls):
        return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


@define
class SigPrice:
    pc_from: Decimal
    mins_since: float
    block_num: int


@define
class AnalysisVolume:
    b_24h: Decimal
    s_24h: Decimal
    b_12h: Decimal
    s_12h: Decimal
    b_6h: Decimal
    s_6h: Decimal
    b_2h: Decimal
    s_2h: Decimal
    b_90m: Decimal
    s_90m: Decimal
    b_60m: Decimal
    s_60m: Decimal
    b_50m: Decimal
    s_50m: Decimal
    b_40m: Decimal
    s_40m: Decimal
    b_30m: Decimal
    s_30m: Decimal
    b_20m: Decimal
    s_20m: Decimal
    b_10m: Decimal
    s_10m: Decimal
    b_40b: Decimal
    s_40b: Decimal
    b_30b: Decimal
    s_30b: Decimal
    b_25b: Decimal
    s_25b: Decimal
    b_20b: Decimal
    s_20b: Decimal
    b_15b: Decimal
    s_15b: Decimal
    b_10b: Decimal
    s_10b: Decimal
    b_8b: Decimal
    s_8b: Decimal
    b_6b: Decimal
    s_6b: Decimal
    b_5b: Decimal
    s_5b: Decimal
    b_4b: Decimal
    s_4b: Decimal
    b_3b: Decimal
    s_3b: Decimal
    b_2b: Decimal
    s_2b: Decimal
    b_1b: Decimal
    s_1b: Decimal


@define
class PairAnalysis:
    pair_addr: Hex
    block_num: int

    holders: tuple[Holder, ...]

    first_block_dt: datetime
    current_dt: datetime

    total_holder_cnt: int
    active_holder_cnt: int
    nosell_holder_cnt: int
    nobuy_holder_cnt: int
    sniper_total_holder_cnt: int
    sniper_active_holder_cnt: int
    sniper_nosell_holder_cnt: int

    median_unreal_amt: Decimal
    liq_pool_quote_amt: Decimal
    market_cap_quote_amt: Decimal

    total_unreal_pnl: Decimal
    total_unreal_amt: Decimal
    total_unreal_profit: Decimal
    total_real_pnl: Decimal
    total_real_profit: Decimal

    snipe_qt_spent_sniping: Decimal
    snipe_unreal_qt_amt: Decimal
    snipe_unreal_pnl: Decimal
    snipe_real_profit: Decimal

    spread: ProfitLossSpread
    sniper_spread: ProfitLossSpread

    ath: SigPrice | None
    atl: SigPrice | None
    rh: SigPrice | None
    rl: SigPrice | None
    mrh: SigPrice | None
    mrl: SigPrice | None
    dh: SigPrice | None
    dl: SigPrice | None
    mdh: SigPrice | None
    mdl: SigPrice | None

    vol: AnalysisVolume

    txcnt_24h: int
    txcnt_12h: int
    txcnt_6h: int
    txcnt_2h: int
    txcnt_90m: int
    txcnt_60m: int
    txcnt_50m: int
    txcnt_40m: int
    txcnt_30m: int
    txcnt_20m: int
    txcnt_10m: int
    txcnt_40b: int
    txcnt_30b: int
    txcnt_25b: int
    txcnt_20b: int
    txcnt_15b: int
    txcnt_10b: int
    txcnt_8b: int
    txcnt_6b: int
    txcnt_5b: int
    txcnt_4b: int
    txcnt_3b: int
    txcnt_2b: int
    txcnt_1b: int

    nhc_24h: int
    nhc_12h: int
    nhc_6h: int
    nhc_2h: int
    nhc_90m: int
    nhc_60m: int
    nhc_30m: int
    nhc_10m: int
    nhc_20b: int
    nhc_10b: int
    nhc_5b: int


@frozen
class PriceLandscape:
    _map: dict[int, UniswapV2Calculator]

    def get_price_calc(self, block_num: int) -> UniswapV2Calculator:
        return self._map[block_num]

    def buy(self, block_num: int, raw_qt_amt: int) -> int:
        return self.get_price_calc(block_num).buy(raw_qt_amt)

    def sell(self, block_num: int, raw_base_amt: int) -> int:
        return self.get_price_calc(block_num).sell(raw_base_amt)


@frozen
class HistoricalData:
    transfers: tuple[TokenTransfer, ...]
    swaps: tuple[Swap, ...]
    price_landscape: PriceLandscape
    upto_bn: int

    def get_swaps(self, block_num: int) -> tuple[Swap, ...]:
        if block_num > self.upto_bn:
            raise ValueError("block number exceeds avaliable data")
        return tuple(s for s in self.swaps if s.block_number <= block_num)

    def get_holder_balances(self, block_num: int) -> Mapping[Hex, int]:
        if block_num > self.upto_bn:
            raise ValueError("block number exceeds avaliable data")
        balances: dict[Hex, int] = defaultdict(lambda: 0)
        for t in self.transfers:
            if t.block_num > block_num:
                break
            balances[t.from_addr] -= t.amount
            balances[t.to_addr] += t.amount
        return balances


# CACHE

_cache = CacheManager.factory(ttl=timedelta(hours=24), max_size=1000)


class _Key(enum.Enum):
    GET_BLOCK_DT = enum.auto()
    GET_TOKEN = enum.auto()


@define
class _LockMgmt:
    _lock: threading.Lock
    _map: dict[Hashable, threading.Lock]

    @classmethod
    def create(cls):
        return _LockMgmt(threading.Lock(), defaultdict(threading.Lock))

    def get_lock(self, resource: str, addr: Hex):
        with self._lock:
            return self._map[(resource, addr)]


# CLASS


@frozen
class HybridTool:
    _lock_mgmt: _LockMgmt

    def sync_blocks(self, upto_bn: int | None) -> None:
        db = get_db()
        rpc = get_rpc()

        if upto_bn is None:
            upto_bn = rpc.get_latest_block_num()

        latest_db_bn = db.get_latest_db_block_num()
        if upto_bn <= latest_db_bn:
            return

        for blocks in rpc.iter_blocks((latest_db_bn + 1, upto_bn)):
            db.insert_blocks(
                [
                    (block.number, datetime.fromtimestamp(block.timestamp, tz=TZ_EST))
                    for block in blocks
                ]
            )

    @_cache.method(_Key.GET_BLOCK_DT, max=150_000)
    def get_block_dt(self, block_num: int) -> datetime:
        db = get_db()
        self.sync_blocks(block_num)
        dt = db.get_block_ts(block_num)
        assert dt
        return dt

    def get_closest_block(self, dt: datetime) -> int:
        block_num, block_ts = get_db().get_nearest_block(dt)
        dif = dt - block_ts
        if dif > timedelta(seconds=30):
            raise RuntimeError(f"block too far away: {round(dif.total_seconds())} sec")
        return block_num

    @_cache.method(_Key.GET_TOKEN)
    def get_token(self, token_addr: Hex) -> Token:
        db = get_db()
        token = db.get_token(token_addr)
        if token:
            return token
        rpc = get_rpc()
        erc_info = rpc.get_erc20_info(token_addr)
        contract_info = get_contract_info(token_addr)
        tx = rpc.get_tx(contract_info.txHash)
        assert tx
        for dbtx in start_tx():
            with dbtx:
                dbtx.insert_token(
                    addr=token_addr,
                    ticker=erc_info.symbol,
                    name=erc_info.name,
                    decimals=erc_info.decimals,
                    creator_addr=contract_info.contractAddress,
                    first_tx_hash=contract_info.txHash,
                    first_block_num=tx.blockNumber,
                )
        token = db.get_token(token_addr)
        assert token
        return token

    def get_token_pair(self, pair_addr: Hex) -> TokenPair:
        db = get_db()
        pair = db.get_pair(pair_addr)
        if pair:
            return pair

        rpc = get_rpc()
        pair_info = rpc.get_univ2_pair_info(pair_addr)
        contract_info = get_contract_info(pair_addr)
        contract_tx = rpc.get_tx_reciept(contract_info.txHash)
        assert contract_tx
        first_block_num = contract_tx.blockNumber

        base_token_addr = pair_info.base_token_addr
        quote_token_addr = pair_info.quote_token_addr
        if pair_info.base_token_addr == WETH_ADDR:
            base_token_addr = pair_info.quote_token_addr
            quote_token_addr = pair_info.base_token_addr

        assert quote_token_addr == WETH_ADDR

        base_token = self.get_token(base_token_addr)
        quote_token = self.get_token(quote_token_addr)

        for tx in start_tx():
            with tx:
                tx.insert_pair(
                    pair_addr=pair_addr,
                    base_addr=base_token.addr,
                    quote_addr=quote_token.addr,
                    first_block_num=first_block_num,
                )

        pair = db.get_pair(pair_addr)
        assert pair
        return pair

    def _sync_transfers(self, token_addr: Hex, end_block_num: int) -> None:
        with self._lock_mgmt.get_lock("transfers", token_addr):
            self._sync_transfers_locked(token_addr, end_block_num)

    def _sync_transfers_locked(self, token_addr: Hex, end_block_num: int) -> None:
        db = get_db()
        prev_block_num = db.get_latest_transfer_block_num(token_addr)
        if end_block_num <= prev_block_num:
            return

        rpc = get_rpc()

        start_log_bn = max(MIN_BLOCK_NUM, prev_block_num + 1)

        for log_chunk in rpc.iter_logs_cryo(
            block_range=(start_log_bn, end_block_num),
            contract_addr=token_addr,
            topic0=TRANSFER_SIG.force_32b(),
        ):
            block_to_transfers: dict[int, list[TransferEvent]] = defaultdict(list)
            for log in log_chunk:
                te = TransferEvent.from_cryo_log(log)
                tx = rpc.get_tx(te.tx_hash)
                assert tx
                block_to_transfers[tx.blockNumber].append(te)

            block_transfers_pairs = list(block_to_transfers.items())
            block_transfers_pairs.sort(key=lambda btp: btp[0])
            for block_num, transfers in block_transfers_pairs:
                for dbtx in start_tx():
                    with dbtx:
                        for te in transfers:
                            dbtx.insert_token_transfer(
                                tx_hash=te.tx_hash,
                                token_addr=te.contract_addr,
                                from_addr=te.src_addr,
                                to_addr=te.dest_addr,
                                amount=te.amount,
                                block_num=block_num,
                                log_index=te.log_index,
                            )
                        dbtx.set_transfer_sync_upto_bn(token_addr, block_num)

        for dbtx in start_tx():
            with dbtx:
                dbtx.set_transfer_sync_upto_bn(token_addr, end_block_num)

    def get_token_balance(
        self,
        token_addr: Hex,
        holder_addr: Hex,
        block_num: int | None,
    ) -> int:
        if block_num is None:
            return get_rpc().get_erc20_balance(token_addr, holder_addr)
        self._sync_transfers(token_addr, block_num)
        return get_db().get_token_balance(token_addr, holder_addr, block_num)

    def get_all_holder_balances(
        self, token_addr: Hex, block_num: int
    ) -> Mapping[Hex, int]:
        self._sync_transfers(token_addr, block_num)
        return get_db().get_all_holder_balances(token_addr, block_num)

    def _sync_swaps(self, pair_addr: Hex, target_block: int) -> None:
        with self._lock_mgmt.get_lock("swaps", pair_addr):
            self._sync_swaps_locked(pair_addr, target_block)

    def _sync_swaps_locked(self, pair_addr: Hex, target_block: int) -> None:
        self.sync_blocks(target_block)

        rpc = get_rpc()
        db = get_db()

        # call this first to ensure pair exists
        pair = self.get_token_pair(pair_addr)
        base_token_addr = pair.base_token_addr

        latest_synced_block = db.get_latest_swap_block_num(pair_addr)

        if latest_synced_block >= target_block:
            return

        swap_sig = _get_event_sig(
            "Swap(address,uint256,uint256,uint256,uint256,address)"
        )
        transfer_sig = _get_event_sig("Transfer(address,address,uint256)")

        for logs in rpc.iter_logs_cryo(
            block_range=(latest_synced_block + 1, target_block),
            contract_addr=pair.addr,
            topic0=swap_sig,
        ):
            swaps_per_tx: dict[Hex, set[SwapEvent]] = defaultdict(set)
            for log in logs:
                swap = SwapEvent.from_cryo_log(log)
                swaps_per_tx[swap.tx_hash].add(swap)

            bn_to_hlswaps: dict[int, set[HighLevelSwap]] = defaultdict(set)
            for tx_hash, swaps in swaps_per_tx.items():
                receipt = rpc.get_tx_reciept(tx_hash)
                if receipt is None:
                    continue
                transfers: list[TransferEvent] = []
                for rlog in receipt.logs:
                    if rlog.get_topic(0) != transfer_sig:
                        continue
                    transfer = TransferEvent.from_log(rlog)
                    transfers.append(transfer)

                base_amt = 0
                quote_amt = 0

                # use swap logs to determine quote amt
                # due to taxes, swap logs will be innaccurate for base amts
                for swap in swaps:
                    if swap.contract_addr != pair.addr:
                        continue
                    mod = 1
                    amount0 = swap.amount0_out - swap.amount0_in
                    amount1 = swap.amount1_out - swap.amount1_in
                    abs_amount0 = abs(amount0)
                    abs_amount1 = abs(amount1)

                    for transfer in transfers:
                        if transfer.contract_addr != pair.quote_token_addr:
                            continue
                        if transfer.amount == abs_amount0:
                            quote_amt += amount0
                            break
                        if transfer.amount == abs_amount1:
                            quote_amt += amount1
                            break

                for transfer in transfers:
                    if transfer.contract_addr != base_token_addr:
                        continue
                    if transfer.dest_addr == base_token_addr:
                        # this is a tax transfer, skip
                        continue
                    if transfer.src_addr == transfer.dest_addr:
                        # this is an internal transfer, skip
                        continue
                    mod = 1 if quote_amt < 0 else -1

                    base_amt += mod * transfer.amount

                if base_amt == 0 and quote_amt == 0:
                    # most likely a sandwich attack, just skip
                    continue

                hl_swap = HighLevelSwap(
                    pair_addr=pair.addr,
                    block_number=receipt.blockNumber,
                    tx_hash=tx_hash,
                    maker_addr=receipt.from_addr,
                    base_amt_change=base_amt,
                    quote_amt_change=quote_amt,
                )
                bn_to_hlswaps[hl_swap.block_number].add(hl_swap)

            bn_hlswaps_pairs = list(bn_to_hlswaps.items())
            bn_hlswaps_pairs.sort(key=lambda bhp: bhp[0])
            for block_num, hl_swaps in bn_hlswaps_pairs:
                for tx in start_tx():
                    with tx:
                        for hl_swap in hl_swaps:
                            tx.insert_swap(
                                tx_hash=hl_swap.tx_hash,
                                pair_addr=hl_swap.pair_addr,
                                actor_addr=hl_swap.maker_addr,
                                block_num=hl_swap.block_number,
                                base_amt_change=hl_swap.base_amt_change,
                                quote_amt_change=hl_swap.quote_amt_change,
                            )
                        tx.set_swap_sync_upto_bn(pair_addr, block_num)

        for tx in start_tx():
            with tx:
                tx.set_swap_sync_upto_bn(pair_addr, target_block)
        return

    def get_swaps(self, pair_addr: Hex, upto_block: int | None = None):
        upto_block = upto_block or get_rpc().get_latest_block_num()
        self._sync_swaps(pair_addr, upto_block)
        return get_db().get_swaps(pair_addr, upto_block)

    def get_uniswap_v2_price_calc(self, pair_addr: Hex, block_num: int | None):
        pair = self.get_token_pair(pair_addr)
        base_amt: int
        quote_amt: int
        if block_num is None:
            rpc = get_rpc()
            base_amt = rpc.get_erc20_balance(pair.base_token_addr, pair_addr)
            quote_amt = rpc.get_erc20_balance(pair.quote_token_addr, pair_addr)
        else:
            base_amt = self.get_token_balance(
                token_addr=pair.base_token_addr,
                holder_addr=pair_addr,
                block_num=block_num,
            )
            quote_amt = self.get_token_balance(
                token_addr=pair.quote_token_addr,
                holder_addr=pair_addr,
                block_num=block_num,
            )
        return UniswapV2Calculator.create(base_amt, quote_amt)

    def get_historical_data(self, pair_addr: Hex, upto_bn: int):
        pair = self.get_token_pair(pair_addr)

        self._sync_transfers(pair.base_token_addr, upto_bn)
        self._sync_transfers(pair.quote_token_addr, upto_bn)

        db = get_db()

        calc_map: dict[int, UniswapV2Calculator] = {}
        for br in split_range(pair.first_block_num, upto_bn, 250):
            for pb in db.get_pair_balances(pair_addr, br[0], br[1]):
                calc = UniswapV2Calculator.create(pb.base_balance, pb.quote_balance)
                calc_map[pb.block_number] = calc
        pl = PriceLandscape(calc_map)

        transfers = db.get_all_transfers(pair.base_token_addr, upto_bn)

        self._sync_swaps(pair_addr, upto_bn)
        swaps = self.get_swaps(pair_addr, upto_bn)

        return HistoricalData(
            transfers=transfers,
            swaps=swaps,
            price_landscape=pl,
            upto_bn=upto_bn,
        )

    def get_watched_pairs(self) -> tuple[TokenPair, ...]:
        now = datetime.now(tz=TZ_EST)
        db = get_db()
        watched_pairs: list[TokenPair] = []
        for pair_addr in db.get_watched_pairs():
            first_swap_bn = self.get_first_swap_bn(pair_addr)
            if first_swap_bn is None:
                continue
            since_launch = now - self.get_block_dt(first_swap_bn)
            if timedelta(hours=53) < since_launch:
                db.set_pair_watching(pair_addr, False)
                continue
            pair = self.get_token_pair(pair_addr)
            watched_pairs.append(pair)
        return tuple(watched_pairs)

    def get_pair_state(
        self,
        pair_addr: Hex,
        at_block: int | None,
        hd: HistoricalData | None = None,
    ) -> PairState | None:
        rpc = get_rpc()
        pair = self.get_token_pair(pair_addr)
        base_token = self.get_token(pair.base_token_addr)

        is_latest = False
        if at_block is None:
            is_latest = True
            at_block = rpc.get_latest_block_num()

        swaps = hd.get_swaps(at_block) if hd else self.get_swaps(pair_addr, at_block)
        holder_balances = (
            hd.get_holder_balances(at_block)
            if hd
            else self.get_all_holder_balances(pair.base_token_addr, at_block)
        )

        if len(swaps) == 0:
            return None

        overall_first_buy_block = swaps[0].block_number

        def is_snipe_tx(block_num: int) -> bool:
            return (block_num - overall_first_buy_block) < 10

        swaps_per_holder: dict[Hex, list[Swap]] = defaultdict(list)
        for swap in swaps:
            swaps_per_holder[swap.maker_addr].append(swap)
        for holder_addr in holder_balances.keys():
            if holder_addr in swaps_per_holder:
                continue
            swaps_per_holder[holder_addr] = []

        now = self.get_block_dt(at_block)
        pair_age = now - self.get_block_dt(pair.first_block_num)
        max_dif = timedelta(minutes=25)
        if pair_age > timedelta(hours=6):
            max_dif = timedelta(minutes=180)

        def to_holder(holder_addr: Hex, holder_swaps: Sequence[Swap]) -> Holder | None:
            if holder_addr == pair_addr:
                return None
            if holder_addr == pair.base_token_addr:
                return None
            if holder_addr in BLACKLISTED_ADDRS:
                return None
            quote_amt_spent_sniping: int = 0
            holding_amt: int = 0
            avg_price: Decimal = Decimal(0)
            gave: int = 0
            recieved: int = 0
            last_action: tuple[str, datetime] | None = None
            buy_blocks: set[int] = set()
            sell_blocks: set[int] = set()
            first_buy_block: int | None = None
            first_buy_ts: datetime | None = None
            has_sold = False
            for swap in holder_swaps:
                swap_ts = self.get_block_dt(swap.block_number)
                if swap.base_amt_change > 0 and swap.quote_amt_change < 0:
                    # BUY
                    gave += abs(swap.quote_amt_change)
                    new_holding_amt = holding_amt + swap.base_amt_change
                    avg_price = (
                        avg_price * holding_amt + swap.price * swap.base_amt_change
                    ) / new_holding_amt
                    holding_amt = new_holding_amt
                    last_action = ("[green]buy ", swap_ts)
                    buy_blocks.add(swap.block_number)
                    if first_buy_block is None:
                        first_buy_block = swap.block_number
                        first_buy_ts = swap_ts
                    if is_snipe_tx(swap.block_number):
                        quote_amt_spent_sniping += abs(swap.quote_amt_change)
                elif swap.base_amt_change < 0 and swap.quote_amt_change > 0:
                    # SELL
                    recieved += swap.quote_amt_change
                    holding_amt = max(0, holding_amt + swap.base_amt_change)
                    last_action = ("[red]sell", swap_ts)
                    sell_blocks.add(swap.block_number)
                    has_sold = True
                else:
                    # raise RuntimeError(f"weird tx: {swap.tx_hash}")
                    continue

            if len(buy_blocks) > 0 and buy_blocks == sell_blocks:
                # ignore sandwich bots
                return None

            holding_amt = holder_balances.get(holder_addr, 0)
            if holding_amt < 0:
                if holder_addr == base_token.creator_addr:
                    holding_amt = 0
                else:
                    raise RuntimeError(f"holder has negative balance: {holder_addr}")

            orig_quote_amt = math.floor(holding_amt * avg_price)
            realized_pnl = recieved + orig_quote_amt - gave

            last_action_desc = ""
            if last_action:
                dif = now - last_action[1]
                if dif < max_dif:
                    dif_mins = round(dif.total_seconds() / 60)
                    last_action_desc = f"{last_action[0]} {dif_mins}m"

            is_sniper = bool(first_buy_block and is_snipe_tx(first_buy_block))

            first_buy: HolderFirstBuy | None = None
            if first_buy_block and first_buy_ts:
                first_buy = HolderFirstBuy(
                    bn=first_buy_block,
                    ts=first_buy_ts,
                )

            return Holder(
                addr=holder_addr,
                holding_amt=holding_amt,
                avg_buy_price=avg_price,
                quote_amt_given=gave,
                quote_amt_recieved=recieved,
                realized_pnl=realized_pnl,
                unrealized_pnl=0,
                unrealized_quote_amt=0,
                unrealized_quote_percent=0,
                total_pnl=0,
                is_active=False,
                last_action=last_action_desc,
                is_sniper=is_sniper,
                has_sold=has_sold,
                quote_amt_spent_sniping=quote_amt_spent_sniping,
                fbuy=first_buy,
            )

        h_results = [
            to_holder(holder_addr, holder_swaps)
            for holder_addr, holder_swaps in swaps_per_holder.items()
        ]
        holders = [h for h in h_results if h is not None]

        price_calc: UniswapV2Calculator = (
            hd.price_landscape.get_price_calc(at_block)
            if hd
            else self.get_uniswap_v2_price_calc(
                pair_addr=pair.addr,
                block_num=None if is_latest else at_block,
            )
        )

        weth_bound = 0.0075 * (10**18)

        total_unrealized_quote = 0
        for holder in holders:
            orig_quote_amt = round(holder.holding_amt * holder.avg_buy_price)
            unrealized_quote_amt = price_calc.sell(holder.holding_amt)
            total_unrealized_quote += unrealized_quote_amt
            holder.unrealized_quote_amt = unrealized_quote_amt
            holder.unrealized_pnl = unrealized_quote_amt - orig_quote_amt
            holder.is_active = unrealized_quote_amt > weth_bound
            holder.total_pnl = holder.realized_pnl + holder.unrealized_pnl

        for holder in holders:
            if not holder.is_active:
                continue
            holder.unrealized_quote_percent = round(
                number=100 * holder.unrealized_quote_amt / total_unrealized_quote,
                ndigits=1,
            )

        return PairState(
            block_num=at_block,
            block_dt=self.get_block_dt(at_block),
            quote_token=self.get_token(pair.quote_token_addr),
            swaps=swaps,
            holders=tuple(holders),
            price_calc=price_calc,
            first_swap_block_num=overall_first_buy_block,
            first_swap_block_dt=self.get_block_dt(overall_first_buy_block),
        )

    def analyze_pair(
        self,
        pair_addr: Hex,
        block_num: int | None,
        hd: HistoricalData | None = None,
    ) -> PairAnalysis | None:
        pair_state = self.get_pair_state(pair_addr, block_num, hd)
        if pair_state is None:
            return None

        block_num = pair_state.block_num
        block_dt = pair_state.block_dt
        holders = pair_state.holders
        first_swap_bn = pair_state.first_swap_block_num
        first_swap_block_dt = pair_state.first_swap_block_dt
        quote_token = pair_state.quote_token
        swaps = pair_state.swaps
        price_calc = pair_state.price_calc

        ap = [h for h in holders if h.is_active]
        total_unreal_pnl = sum(h.unrealized_pnl for h in ap)
        total_unreal_quote = sum(h.unrealized_quote_amt for h in ap)
        total_unreal_profit = sum(h.unrealized_pnl for h in ap if h.unrealized_pnl > 0)
        total_real_pnl = sum(h.realized_pnl for h in holders)
        total_real_profit = sum(h.realized_pnl for h in ap if h.realized_pnl > 0)

        median_unreal_quote = (
            statistics.median_low([h.unrealized_quote_amt for h in ap])
            if len(ap) > 0
            else 0
        )

        snipers = list(h for h in holders if h.is_sniper)
        snipe_quote_amt_spent = sum(h.quote_amt_spent_sniping for h in snipers)
        snipe_unreal_qt_amt = price_calc.sell(sum(h.holding_amt for h in snipers))
        snipe_unreal_pnl = snipe_unreal_qt_amt - snipe_quote_amt_spent
        snipe_real_profit = sum(h.realized_pnl for h in snipers if h.realized_pnl > 0)

        def to_volume(swaps: Iterable[Swap]) -> tuple[int, int]:
            buy_vol: int = 0
            sell_vol: int = 0
            for s in swaps:
                abs_qt_amt = abs(s.quote_amt_change)
                if s.base_amt_change > 0:
                    buy_vol += abs_qt_amt
                elif s.base_amt_change < 0:
                    sell_vol += abs_qt_amt
            return buy_vol, sell_vol

        def gen_spread(holders: Iterable[Holder]):
            spread = ProfitLossSpread.create_empty()
            for holder in holders:
                qp = holder.unrealized_quote_percent
                upc = holder.get_unreal_pc()
                if upc >= 500:
                    spread.p500 += qp
                if upc >= 400:
                    spread.p400 += qp
                if upc >= 300:
                    spread.p300 += qp
                if upc >= 250:
                    spread.p250 += qp
                if upc >= 200:
                    spread.p200 += qp
                if upc >= 150:
                    spread.p150 += qp
                if upc >= 100:
                    spread.p100 += qp
                if upc >= 80:
                    spread.p80 += qp
                if upc >= 60:
                    spread.p60 += qp
                if upc >= 40:
                    spread.p40 += qp
                if upc >= 20:
                    spread.p20 += qp
                if upc > -20:
                    spread.breakeven += qp
                if upc > -40:
                    spread.l20 += qp
                if upc > -60:
                    spread.l40 += qp
                if upc > -80:
                    spread.l60 += qp
                spread.l80 += qp
            return spread

        def to_dec(raw_amt: int):
            return round(quote_token.raw_to_dec(raw_amt), 4)

        def find_sig_price(
            swaps: Sequence[Swap],
            mode: Literal["peak", "valley"],
            prev_price: SigPrice | None,
            direction: Literal["forward", "backward"] = "forward",
            buffer: int = 15,
        ) -> SigPrice | None:
            min_bn = first_swap_bn + buffer
            max_bn = block_num
            if prev_price:
                if direction == "forward":
                    bn_dif = block_num - prev_price.block_num
                    min_bn = max(
                        min_bn,
                        math.floor(prev_price.block_num + bn_dif * 0.15),
                    )
                else:
                    bn_dif = prev_price.block_num - min_bn
                    max_bn = min(
                        max_bn,
                        math.floor(prev_price.block_num - bn_dif * 0.15),
                    )
            if max_bn <= min_bn:
                return prev_price
            fswaps = [
                s
                for s in swaps
                if min_bn <= s.block_number <= max_bn and s.quote_amt_change != 0
            ]
            if len(fswaps) == 0:
                return prev_price
            fswaps.sort(key=lambda s: s.price, reverse=mode == "peak")
            s = fswaps[0]
            time_dif = block_dt - self.get_block_dt(s.block_number)
            orig_amt = abs(s.quote_amt_change)
            new_amt = price_calc.sell(abs(s.base_amt_change))
            return SigPrice(
                pc_from=percent_change(orig_amt, new_amt),
                mins_since=time_dif.total_seconds() / 60,
                block_num=s.block_number,
            )

        small_eth_amt = 0.2 * (10**18)
        small_buys: list[Swap] = []
        small_sells: list[Swap] = []
        for s in swaps:
            if small_eth_amt < abs(s.quote_amt_change):
                continue
            if s.base_amt_change > 0:
                small_buys.append(s)
            elif s.base_amt_change < 0:
                small_sells.append(s)

        ath = find_sig_price(small_buys, "peak", None)
        atl = find_sig_price(small_sells, "valley", None)
        rh = find_sig_price(small_buys, "peak", ath)
        rl = find_sig_price(small_sells, "valley", atl)
        mrh = find_sig_price(small_buys, "peak", rh)
        mrl = find_sig_price(small_sells, "valley", rl)
        dh = find_sig_price(small_buys, "peak", ath, "backward")
        dl = find_sig_price(small_sells, "valley", atl, "backward")
        mdh = find_sig_price(small_buys, "peak", dh, "backward", 0)
        mdl = find_sig_price(small_sells, "valley", dl, "backward", 0)

        dt_24h_ago = block_dt - timedelta(hours=24)
        dt_12h_ago = block_dt - timedelta(hours=12)
        dt_6h_ago = block_dt - timedelta(hours=6)
        dt_2h_ago = block_dt - timedelta(hours=2)
        dt_90m_ago = block_dt - timedelta(minutes=90)
        dt_60m_ago = block_dt - timedelta(minutes=60)
        dt_50m_ago = block_dt - timedelta(minutes=50)
        dt_40m_ago = block_dt - timedelta(minutes=40)
        dt_30m_ago = block_dt - timedelta(minutes=30)
        dt_20m_ago = block_dt - timedelta(minutes=20)
        dt_10m_ago = block_dt - timedelta(minutes=10)

        nh_24h = [h for h in holders if h.fbuy and h.fbuy.ts >= dt_24h_ago]
        nh_12h = [h for h in nh_24h if h.fbuy and h.fbuy.ts >= dt_12h_ago]
        nh_6h = [h for h in nh_12h if h.fbuy and h.fbuy.ts >= dt_6h_ago]
        nh_2h = [h for h in nh_6h if h.fbuy and h.fbuy.ts >= dt_2h_ago]
        nh_90m = [h for h in nh_2h if h.fbuy and h.fbuy.ts >= dt_90m_ago]
        nh_60m = [h for h in nh_90m if h.fbuy and h.fbuy.ts >= dt_60m_ago]
        nh_30m = [h for h in nh_60m if h.fbuy and h.fbuy.ts >= dt_30m_ago]
        nh_10m = [h for h in nh_30m if h.fbuy and h.fbuy.ts >= dt_10m_ago]
        nh_20b = [h for h in nh_10m if h.fbuy and h.fbuy.bn >= block_num - 20]
        nh_10b = [h for h in nh_20b if h.fbuy and h.fbuy.bn >= block_num - 10]
        nh_5b = [h for h in nh_10b if h.fbuy and h.fbuy.bn >= block_num - 5]

        swaps_24h = [s for s in swaps if s.timestamp >= dt_24h_ago]
        swaps_12h = [s for s in swaps_24h if s.timestamp >= dt_12h_ago]
        swaps_6h = [s for s in swaps_12h if s.timestamp >= dt_6h_ago]
        swaps_2h = [s for s in swaps_6h if s.timestamp >= dt_2h_ago]
        swaps_90m = [s for s in swaps_2h if s.timestamp >= dt_90m_ago]
        swaps_60m = [s for s in swaps_90m if s.timestamp >= dt_60m_ago]
        swaps_50m = [s for s in swaps_60m if s.timestamp >= dt_50m_ago]
        swaps_40m = [s for s in swaps_50m if s.timestamp >= dt_40m_ago]
        swaps_30m = [s for s in swaps_40m if s.timestamp >= dt_30m_ago]
        swaps_20m = [s for s in swaps_30m if s.timestamp >= dt_20m_ago]
        swaps_10m = [s for s in swaps_20m if s.timestamp >= dt_10m_ago]
        swaps_40b = [s for s in swaps_10m if s.block_number >= block_num - 40]
        swaps_30b = [s for s in swaps_40b if s.block_number >= block_num - 30]
        swaps_25b = [s for s in swaps_30b if s.block_number >= block_num - 25]
        swaps_20b = [s for s in swaps_25b if s.block_number >= block_num - 20]
        swaps_15b = [s for s in swaps_20b if s.block_number >= block_num - 15]
        swaps_10b = [s for s in swaps_15b if s.block_number >= block_num - 10]
        swaps_8b = [s for s in swaps_10b if s.block_number >= block_num - 8]
        swaps_6b = [s for s in swaps_8b if s.block_number >= block_num - 6]
        swaps_5b = [s for s in swaps_6b if s.block_number >= block_num - 5]
        swaps_4b = [s for s in swaps_5b if s.block_number >= block_num - 4]
        swaps_3b = [s for s in swaps_4b if s.block_number >= block_num - 3]
        swaps_2b = [s for s in swaps_3b if s.block_number >= block_num - 2]
        swaps_1b = [s for s in swaps_2b if s.block_number >= block_num - 1]

        b_vol_24h, s_vol_24h = to_volume(swaps_24h)
        b_vol_12h, s_vol_12h = to_volume(swaps_12h)
        b_vol_6h, s_vol_6h = to_volume(swaps_6h)
        b_vol_2h, s_vol_2h = to_volume(swaps_2h)
        b_vol_90m, s_vol_90m = to_volume(swaps_90m)
        b_vol_60m, s_vol_60m = to_volume(swaps_60m)
        b_vol_50m, s_vol_50m = to_volume(swaps_50m)
        b_vol_40m, s_vol_40m = to_volume(swaps_40m)
        b_vol_30m, s_vol_30m = to_volume(swaps_30m)
        b_vol_20m, s_vol_20m = to_volume(swaps_20m)
        b_vol_10m, s_vol_10m = to_volume(swaps_10m)
        b_vol_40b, s_vol_40b = to_volume(swaps_40b)
        b_vol_30b, s_vol_30b = to_volume(swaps_30b)
        b_vol_25b, s_vol_25b = to_volume(swaps_25b)
        b_vol_20b, s_vol_20b = to_volume(swaps_20b)
        b_vol_15b, s_vol_15b = to_volume(swaps_15b)
        b_vol_10b, s_vol_10b = to_volume(swaps_10b)
        b_vol_8b, s_vol_8b = to_volume(swaps_8b)
        b_vol_6b, s_vol_6b = to_volume(swaps_6b)
        b_vol_5b, s_vol_5b = to_volume(swaps_5b)
        b_vol_4b, s_vol_4b = to_volume(swaps_4b)
        b_vol_3b, s_vol_3b = to_volume(swaps_3b)
        b_vol_2b, s_vol_2b = to_volume(swaps_2b)
        b_vol_1b, s_vol_1b = to_volume(swaps_1b)
        vol = AnalysisVolume(
            b_24h=to_dec(b_vol_24h),
            s_24h=to_dec(s_vol_24h),
            b_12h=to_dec(b_vol_12h),
            s_12h=to_dec(s_vol_12h),
            b_6h=to_dec(b_vol_6h),
            s_6h=to_dec(s_vol_6h),
            b_2h=to_dec(b_vol_2h),
            s_2h=to_dec(s_vol_2h),
            b_90m=to_dec(b_vol_90m),
            s_90m=to_dec(s_vol_90m),
            b_60m=to_dec(b_vol_60m),
            s_60m=to_dec(s_vol_60m),
            b_50m=to_dec(b_vol_50m),
            s_50m=to_dec(s_vol_50m),
            b_40m=to_dec(b_vol_40m),
            s_40m=to_dec(s_vol_40m),
            b_30m=to_dec(b_vol_30m),
            s_30m=to_dec(s_vol_30m),
            b_20m=to_dec(b_vol_20m),
            s_20m=to_dec(s_vol_20m),
            b_10m=to_dec(b_vol_10m),
            s_10m=to_dec(s_vol_10m),
            b_40b=to_dec(b_vol_40b),
            s_40b=to_dec(s_vol_40b),
            b_30b=to_dec(b_vol_30b),
            s_30b=to_dec(s_vol_30b),
            b_25b=to_dec(b_vol_25b),
            s_25b=to_dec(s_vol_25b),
            b_20b=to_dec(b_vol_20b),
            s_20b=to_dec(s_vol_20b),
            b_15b=to_dec(b_vol_15b),
            s_15b=to_dec(s_vol_15b),
            b_10b=to_dec(b_vol_10b),
            s_10b=to_dec(s_vol_10b),
            b_8b=to_dec(b_vol_8b),
            s_8b=to_dec(s_vol_8b),
            b_6b=to_dec(b_vol_6b),
            s_6b=to_dec(s_vol_6b),
            b_5b=to_dec(b_vol_5b),
            s_5b=to_dec(s_vol_5b),
            b_4b=to_dec(b_vol_4b),
            s_4b=to_dec(s_vol_4b),
            b_3b=to_dec(b_vol_3b),
            s_3b=to_dec(s_vol_3b),
            b_2b=to_dec(b_vol_2b),
            s_2b=to_dec(s_vol_2b),
            b_1b=to_dec(b_vol_1b),
            s_1b=to_dec(s_vol_1b),
        )

        market_cap_qt_amt = round(
            price_calc.get_price(MKTCAP_ETH_AMT) * sum(h.holding_amt for h in holders)
        )

        return PairAnalysis(
            pair_addr=pair_addr,
            block_num=block_num,
            # s
            holders=holders,
            # s
            total_holder_cnt=len(holders),
            active_holder_cnt=len(ap),
            nosell_holder_cnt=len([h for h in holders if not h.has_sold]),
            nobuy_holder_cnt=len([h for h in holders if h.fbuy is None]),
            sniper_total_holder_cnt=len(snipers),
            sniper_active_holder_cnt=len([h for h in snipers if h.is_active]),
            sniper_nosell_holder_cnt=len([h for h in snipers if not h.has_sold]),
            # s
            median_unreal_amt=to_dec(median_unreal_quote),
            liq_pool_quote_amt=to_dec(price_calc.quote_amt),
            market_cap_quote_amt=to_dec(market_cap_qt_amt),
            # s
            first_block_dt=first_swap_block_dt,
            current_dt=block_dt,
            # s
            total_unreal_pnl=to_dec(total_unreal_pnl),
            total_unreal_amt=to_dec(total_unreal_quote),
            total_unreal_profit=to_dec(total_unreal_profit),
            total_real_pnl=to_dec(total_real_pnl),
            total_real_profit=to_dec(total_real_profit),
            # s
            snipe_qt_spent_sniping=to_dec(snipe_quote_amt_spent),
            snipe_unreal_qt_amt=to_dec(snipe_unreal_qt_amt),
            snipe_unreal_pnl=to_dec(snipe_unreal_pnl),
            snipe_real_profit=to_dec(snipe_real_profit),
            # s
            spread=gen_spread(ap),
            sniper_spread=gen_spread(snipers),
            # s
            ath=ath,
            atl=atl,
            rh=rh,
            rl=rl,
            mrh=mrh,
            mrl=mrl,
            dh=dh,
            dl=dl,
            mdh=mdh,
            mdl=mdl,
            # s
            vol=vol,
            # s
            txcnt_24h=len(swaps_24h),
            txcnt_12h=len(swaps_12h),
            txcnt_6h=len(swaps_6h),
            txcnt_2h=len(swaps_2h),
            txcnt_90m=len(swaps_90m),
            txcnt_60m=len(swaps_60m),
            txcnt_50m=len(swaps_50m),
            txcnt_40m=len(swaps_40m),
            txcnt_30m=len(swaps_30m),
            txcnt_20m=len(swaps_20m),
            txcnt_10m=len(swaps_10m),
            txcnt_40b=len(swaps_40b),
            txcnt_30b=len(swaps_30b),
            txcnt_25b=len(swaps_25b),
            txcnt_20b=len(swaps_20b),
            txcnt_15b=len(swaps_15b),
            txcnt_10b=len(swaps_10b),
            txcnt_8b=len(swaps_8b),
            txcnt_6b=len(swaps_6b),
            txcnt_5b=len(swaps_5b),
            txcnt_4b=len(swaps_4b),
            txcnt_3b=len(swaps_3b),
            txcnt_2b=len(swaps_2b),
            txcnt_1b=len(swaps_1b),
            # s
            nhc_24h=len(nh_24h),
            nhc_12h=len(nh_12h),
            nhc_6h=len(nh_6h),
            nhc_2h=len(nh_2h),
            nhc_90m=len(nh_90m),
            nhc_60m=len(nh_60m),
            nhc_30m=len(nh_30m),
            nhc_10m=len(nh_10m),
            nhc_20b=len(nh_20b),
            nhc_10b=len(nh_10b),
            nhc_5b=len(nh_5b),
        )

    def get_first_swap_bn(self, pair_addr: Hex) -> int | None:
        pair = self.get_token_pair(pair_addr)
        if pair is None:
            return None
        latest_bn = get_rpc().get_latest_block_num()
        interval = 100
        upto_bn = pair.first_block_num + interval
        swaps: Sequence[Swap] = []
        while len(swaps) == 0 and upto_bn < latest_bn:
            swaps = self.get_swaps(pair_addr, upto_bn)
            upto_bn += interval
        if len(swaps) == 0:
            return None
        return swaps[0].block_number


_hybrid: HybridTool | None = None
_hybrid_lock = threading.Lock()


def get_hybrid():
    global _hybrid
    with _hybrid_lock:
        if _hybrid is None:
            _hybrid = HybridTool(_LockMgmt.create())
        return _hybrid
