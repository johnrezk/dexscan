import csv
import functools
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Type

import polars as pl
from alive_progress import alive_bar
from attrs import Attribute, asdict, define, fields

from dexscan.constants import PROJECT_DIR
from dexscan.hybrid import HistoricalData, PairAnalysis, PriceLandscape, get_hybrid
from dexscan.primitives import Hex
from dexscan.utils import df_to_json, percent_change

ETH_HAND_SIZE = round(0.1 * (10**18))

HOURS_OF_SAMPLING = 48
SKIP_FIRST_BLOCKS = 10

SCORE_PERIOD_BLKS = 600  # ~2 hours
TOTAL_CLASSES = 3

SAMPLES_DIR = PROJECT_DIR / "samples"
SAMPLES_CSV = PROJECT_DIR / "samples_train.csv"


@define
class InputData:
    pair_addr: str
    block_num: int

    hours_since_midnight: float
    hours_since_launch: float

    total_holder_cnt: float
    active_holder_cnt: float
    nosell_holder_cnt: float
    nobuy_holder_cnt: float
    sniper_total_holder_cnt: float
    sniper_active_holder_cnt: float
    sniper_nosell_holder_cnt: float

    median_unreal_quote_amt: float
    liq_pool_quote_amt: float
    market_cap_quote_amt: float

    total_unreal_pnl: float
    total_unreal_quote_amt: float
    total_unreal_profit: float
    total_real_pnl: float
    total_real_profit: float

    sniper_initial_volume: float
    sniper_unreal_pnl: float
    sniper_real_profit: float

    pcliq_p500: float
    pcliq_p400: float
    pcliq_p300: float
    pcliq_p250: float
    pcliq_p200: float
    pcliq_p150: float
    pcliq_p100: float
    pcliq_p80: float
    pcliq_p60: float
    pcliq_p40: float
    pcliq_p20: float
    pcliq_breakeven: float
    pcliq_l20: float
    pcliq_l40: float
    pcliq_l60: float
    pcliq_l80: float

    sniper_pcliq_p500: float
    sniper_pcliq_p400: float
    sniper_pcliq_p300: float
    sniper_pcliq_p250: float
    sniper_pcliq_p200: float
    sniper_pcliq_p150: float
    sniper_pcliq_p100: float
    sniper_pcliq_p80: float
    sniper_pcliq_p60: float
    sniper_pcliq_p40: float
    sniper_pcliq_p20: float
    sniper_pcliq_breakeven: float
    sniper_pcliq_l20: float
    sniper_pcliq_l40: float
    sniper_pcliq_l60: float
    sniper_pcliq_l80: float

    ath_pc_from: float
    ath_mins_since: float
    atl_pc_from: float
    atl_mins_since: float
    rh_pc_from: float
    rh_mins_since: float
    rl_pc_from: float
    rl_mins_since: float
    mrh_pc_from: float
    mrh_mins_since: float
    mrl_pc_from: float
    mrl_mins_since: float
    dh_pc_from: float
    dh_mins_since: float
    dl_pc_from: float
    dl_mins_since: float
    mdh_pc_from: float
    mdh_mins_since: float
    mdl_pc_from: float
    mdl_mins_since: float

    b_vol_24h: float
    s_vol_24h: float
    b_vol_12h: float
    s_vol_12h: float
    b_vol_6h: float
    s_vol_6h: float
    b_vol_2h: float
    s_vol_2h: float
    b_vol_90m: float
    s_vol_90m: float
    b_vol_60m: float
    s_vol_60m: float
    b_vol_50m: float
    s_vol_50m: float
    b_vol_40m: float
    s_vol_40m: float
    b_vol_30m: float
    s_vol_30m: float
    b_vol_20m: float
    s_vol_20m: float
    b_vol_10m: float
    s_vol_10m: float
    b_vol_40b: float
    s_vol_40b: float
    b_vol_30b: float
    s_vol_30b: float
    b_vol_25b: float
    s_vol_25b: float
    b_vol_20b: float
    s_vol_20b: float
    b_vol_15b: float
    s_vol_15b: float
    b_vol_10b: float
    s_vol_10b: float
    b_vol_8b: float
    s_vol_8b: float
    b_vol_6b: float
    s_vol_6b: float
    b_vol_5b: float
    s_vol_5b: float
    b_vol_4b: float
    s_vol_4b: float
    b_vol_3b: float
    s_vol_3b: float
    b_vol_2b: float
    s_vol_2b: float
    b_vol_1b: float
    s_vol_1b: float

    txcnt_24h: float
    txcnt_12h: float
    txcnt_6h: float
    txcnt_2h: float
    txcnt_90m: float
    txcnt_60m: float
    txcnt_50m: float
    txcnt_40m: float
    txcnt_30m: float
    txcnt_20m: float
    txcnt_10m: float
    txcnt_40b: float
    txcnt_30b: float
    txcnt_25b: float
    txcnt_20b: float
    txcnt_15b: float
    txcnt_10b: float
    txcnt_8b: float
    txcnt_6b: float
    txcnt_5b: float
    txcnt_4b: float
    txcnt_3b: float
    txcnt_2b: float
    txcnt_1b: float

    nhc_24h: float
    nhc_12h: float
    nhc_6h: float
    nhc_2h: float
    nhc_90m: float
    nhc_60m: float
    nhc_30m: float
    nhc_10m: float
    nhc_20b: float
    nhc_10b: float
    nhc_5b: float

    @classmethod
    def from_analysis(cls, a: PairAnalysis) -> "InputData | None":
        now = a.current_dt.astimezone(UTC)
        hours_since_launch = (now - a.first_block_dt).total_seconds() / 3600
        hours_since_midnight = now.hour + (now.minute / 60)
        assert a.vol
        if a.ath is None or a.atl is None:
            return None
        if a.rh is None or a.rl is None:
            return None
        if a.mrh is None or a.mrl is None:
            return None
        if a.dh is None or a.dl is None:
            return None
        if a.mdh is None or a.mdl is None:
            return None
        return InputData(
            pair_addr=str(a.pair_addr),
            block_num=a.block_num,
            hours_since_midnight=hours_since_midnight,
            hours_since_launch=hours_since_launch,
            total_holder_cnt=a.total_holder_cnt,
            active_holder_cnt=a.active_holder_cnt,
            nosell_holder_cnt=a.nosell_holder_cnt,
            nobuy_holder_cnt=a.nobuy_holder_cnt,
            sniper_total_holder_cnt=a.sniper_total_holder_cnt,
            sniper_active_holder_cnt=a.sniper_active_holder_cnt,
            sniper_nosell_holder_cnt=a.sniper_nosell_holder_cnt,
            median_unreal_quote_amt=float(a.median_unreal_amt),
            liq_pool_quote_amt=float(a.liq_pool_quote_amt),
            market_cap_quote_amt=float(a.market_cap_quote_amt),
            sniper_initial_volume=float(a.snipe_qt_spent_sniping),
            sniper_unreal_pnl=float(a.snipe_unreal_pnl),
            sniper_real_profit=float(a.snipe_real_profit),
            total_unreal_pnl=float(a.total_unreal_pnl),
            total_unreal_quote_amt=float(a.total_unreal_amt),
            total_unreal_profit=float(a.total_unreal_profit),
            total_real_pnl=float(a.total_real_pnl),
            total_real_profit=float(a.total_real_profit),
            pcliq_p500=a.spread.p500,
            pcliq_p400=a.spread.p400,
            pcliq_p300=a.spread.p300,
            pcliq_p250=a.spread.p250,
            pcliq_p200=a.spread.p200,
            pcliq_p150=a.spread.p150,
            pcliq_p100=a.spread.p100,
            pcliq_p80=a.spread.p80,
            pcliq_p60=a.spread.p60,
            pcliq_p40=a.spread.p40,
            pcliq_p20=a.spread.p20,
            pcliq_breakeven=a.spread.breakeven,
            pcliq_l20=a.spread.l20,
            pcliq_l40=a.spread.l40,
            pcliq_l60=a.spread.l60,
            pcliq_l80=a.spread.l80,
            sniper_pcliq_p500=a.sniper_spread.p500,
            sniper_pcliq_p400=a.sniper_spread.p400,
            sniper_pcliq_p300=a.sniper_spread.p300,
            sniper_pcliq_p250=a.sniper_spread.p250,
            sniper_pcliq_p200=a.sniper_spread.p200,
            sniper_pcliq_p150=a.sniper_spread.p150,
            sniper_pcliq_p100=a.sniper_spread.p100,
            sniper_pcliq_p80=a.sniper_spread.p80,
            sniper_pcliq_p60=a.sniper_spread.p60,
            sniper_pcliq_p40=a.sniper_spread.p40,
            sniper_pcliq_p20=a.sniper_spread.p20,
            sniper_pcliq_breakeven=a.sniper_spread.breakeven,
            sniper_pcliq_l20=a.sniper_spread.l20,
            sniper_pcliq_l40=a.sniper_spread.l40,
            sniper_pcliq_l60=a.sniper_spread.l60,
            sniper_pcliq_l80=a.sniper_spread.l80,
            ath_pc_from=float(a.ath.pc_from),
            ath_mins_since=float(a.ath.mins_since),
            atl_pc_from=float(a.atl.pc_from),
            atl_mins_since=float(a.atl.mins_since),
            rh_pc_from=float(a.rh.pc_from),
            rh_mins_since=float(a.rh.mins_since),
            rl_pc_from=float(a.rl.pc_from),
            rl_mins_since=float(a.rl.mins_since),
            mrh_pc_from=float(a.mrh.pc_from),
            mrh_mins_since=float(a.mrh.mins_since),
            mrl_pc_from=float(a.mrl.pc_from),
            mrl_mins_since=float(a.mrl.mins_since),
            dh_pc_from=float(a.dh.pc_from),
            dh_mins_since=float(a.dh.mins_since),
            dl_pc_from=float(a.dl.pc_from),
            dl_mins_since=float(a.dl.mins_since),
            mdh_pc_from=float(a.mdh.pc_from),
            mdh_mins_since=float(a.mdh.mins_since),
            mdl_pc_from=float(a.mdl.pc_from),
            mdl_mins_since=float(a.mdl.mins_since),
            b_vol_24h=float(a.vol.b_24h),
            s_vol_24h=float(a.vol.s_24h),
            b_vol_12h=float(a.vol.b_12h),
            s_vol_12h=float(a.vol.s_12h),
            b_vol_6h=float(a.vol.b_6h),
            s_vol_6h=float(a.vol.s_6h),
            b_vol_2h=float(a.vol.b_2h),
            s_vol_2h=float(a.vol.s_2h),
            b_vol_90m=float(a.vol.b_90m),
            s_vol_90m=float(a.vol.s_90m),
            b_vol_60m=float(a.vol.b_60m),
            s_vol_60m=float(a.vol.s_60m),
            b_vol_50m=float(a.vol.b_50m),
            s_vol_50m=float(a.vol.s_50m),
            b_vol_40m=float(a.vol.b_40m),
            s_vol_40m=float(a.vol.s_40m),
            b_vol_30m=float(a.vol.b_30m),
            s_vol_30m=float(a.vol.s_30m),
            b_vol_20m=float(a.vol.b_20m),
            s_vol_20m=float(a.vol.s_20m),
            b_vol_10m=float(a.vol.b_10m),
            s_vol_10m=float(a.vol.s_10m),
            b_vol_40b=float(a.vol.b_40b),
            s_vol_40b=float(a.vol.s_40b),
            b_vol_30b=float(a.vol.b_30b),
            s_vol_30b=float(a.vol.s_30b),
            b_vol_25b=float(a.vol.b_25b),
            s_vol_25b=float(a.vol.s_25b),
            b_vol_20b=float(a.vol.b_20b),
            s_vol_20b=float(a.vol.s_20b),
            b_vol_15b=float(a.vol.b_15b),
            s_vol_15b=float(a.vol.s_15b),
            b_vol_10b=float(a.vol.b_10b),
            s_vol_10b=float(a.vol.s_10b),
            b_vol_8b=float(a.vol.b_8b),
            s_vol_8b=float(a.vol.s_8b),
            b_vol_6b=float(a.vol.b_6b),
            s_vol_6b=float(a.vol.s_6b),
            b_vol_5b=float(a.vol.b_5b),
            s_vol_5b=float(a.vol.s_5b),
            b_vol_4b=float(a.vol.b_4b),
            s_vol_4b=float(a.vol.s_4b),
            b_vol_3b=float(a.vol.b_3b),
            s_vol_3b=float(a.vol.s_3b),
            b_vol_2b=float(a.vol.b_2b),
            s_vol_2b=float(a.vol.s_2b),
            b_vol_1b=float(a.vol.b_1b),
            s_vol_1b=float(a.vol.s_1b),
            txcnt_24h=float(a.txcnt_24h),
            txcnt_12h=float(a.txcnt_12h),
            txcnt_6h=float(a.txcnt_6h),
            txcnt_2h=float(a.txcnt_2h),
            txcnt_90m=float(a.txcnt_90m),
            txcnt_60m=float(a.txcnt_60m),
            txcnt_50m=float(a.txcnt_50m),
            txcnt_40m=float(a.txcnt_40m),
            txcnt_30m=float(a.txcnt_30m),
            txcnt_20m=float(a.txcnt_20m),
            txcnt_10m=float(a.txcnt_10m),
            txcnt_40b=float(a.txcnt_40b),
            txcnt_30b=float(a.txcnt_30b),
            txcnt_25b=float(a.txcnt_25b),
            txcnt_20b=float(a.txcnt_20b),
            txcnt_15b=float(a.txcnt_15b),
            txcnt_10b=float(a.txcnt_10b),
            txcnt_8b=float(a.txcnt_8b),
            txcnt_6b=float(a.txcnt_6b),
            txcnt_5b=float(a.txcnt_5b),
            txcnt_4b=float(a.txcnt_4b),
            txcnt_3b=float(a.txcnt_3b),
            txcnt_2b=float(a.txcnt_2b),
            txcnt_1b=float(a.txcnt_1b),
            nhc_24h=float(a.nhc_24h),
            nhc_12h=float(a.nhc_12h),
            nhc_6h=float(a.nhc_6h),
            nhc_2h=float(a.nhc_2h),
            nhc_90m=float(a.nhc_90m),
            nhc_60m=float(a.nhc_60m),
            nhc_30m=float(a.nhc_30m),
            nhc_10m=float(a.nhc_10m),
            nhc_20b=float(a.nhc_20b),
            nhc_10b=float(a.nhc_10b),
            nhc_5b=float(a.nhc_5b),
        )

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)

    def to_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            data=self.to_dict(),
            schema=get_data_polars_schema(),
        )


@functools.cache
def get_score_polars_schema():
    df_schema: dict[str, Type[pl.Utf8 | pl.Int32]] = {
        "pair_addr": pl.Utf8,
        "block_num": pl.Int32,
        "score": pl.Int32,
    }
    return df_schema


@functools.cache
def get_data_polars_schema():
    input_attrs: tuple[Attribute, ...] = fields(InputData)
    df_schema: dict[str, Type[pl.Float32 | pl.Utf8 | pl.Int32]] = {}
    for ia in input_attrs:
        if ia.type == str:
            df_schema[ia.name] = pl.Utf8
        if ia.type == int:
            df_schema[ia.name] = pl.Int32
        if ia.type == float:
            df_schema[ia.name] = pl.Float32
    return df_schema


def score_to_df(pair_addr: Hex, block_num: int, score: int):
    return pl.DataFrame(
        data={
            "pair_addr": str(pair_addr),
            "block_num": block_num,
            "score": score,
        },
        schema=get_score_polars_schema(),
    )


@define
class ProfitPoint:
    block_num: int
    real_pc: float


@define
class Streak:
    started_bn: int
    ended_bn: int
    length: int
    threshold: int


def get_score(block_num: int, pl: PriceLandscape) -> int | None:
    start_score_bn = block_num
    end_score_bn = start_score_bn + SCORE_PERIOD_BLKS

    base_token_amt = statistics.median_low(
        [pl.buy(bn, ETH_HAND_SIZE) for bn in range(block_num + 2, block_num + 4)]
    )
    if base_token_amt == 0:
        return None

    profit_pts: list[ProfitPoint] = []
    for sbn in range(start_score_bn, end_score_bn + 1):
        qt_profit_amt = pl.sell(sbn, base_token_amt)
        pc_change = percent_change(ETH_HAND_SIZE, qt_profit_amt)
        profit_pts.append(
            ProfitPoint(
                block_num=sbn,
                real_pc=float(round(pc_change, 2)),
            )
        )

    BEAR_CEIL = 60
    BULL_FLOOR = -10
    BULL_PEAK = 45

    # 100 blocks == ~20 mins
    thresh_mlen_dir: tuple[tuple[int, int, Literal["above", "below"]], ...] = (
        (BEAR_CEIL, 550, "below"),
        (BULL_FLOOR, 4, "above"),
        (BULL_PEAK, 50, "above"),
    )

    last_index = len(profit_pts) - 1
    streaks: list[Streak] = []
    counter: dict[int, int | None] = defaultdict(lambda: None)
    for index, pt in enumerate(profit_pts):
        for threshold, min_len, direction in thresh_mlen_dir:
            passes: bool = (
                threshold < pt.real_pc
                if direction == "above"
                else pt.real_pc < threshold
            )
            if passes and counter[threshold] is None:
                counter[threshold] = pt.block_num
            streak_first_bn = counter[threshold]
            if streak_first_bn and (not passes or index == last_index):
                streak_len = pt.block_num - streak_first_bn
                if streak_len >= min_len:
                    streaks.append(
                        Streak(
                            started_bn=streak_first_bn,
                            ended_bn=pt.block_num - 1,
                            length=streak_len,
                            threshold=threshold,
                        )
                    )
                counter[threshold] = None

    bear_upper_bn = start_score_bn + 5
    bear_ceils = [
        s for s in streaks if s.threshold == BEAR_CEIL and s.started_bn < bear_upper_bn
    ]
    if len(bear_ceils) > 0:
        return 2

    bull_upper_bn = start_score_bn + 5
    bull_floors = [
        s for s in streaks if s.threshold == BULL_FLOOR and s.started_bn < bull_upper_bn
    ]
    if len(bull_floors) > 0:
        assert len(bull_floors) == 1
        bull_floor = bull_floors[0]
        bull_peaks = [
            s
            for s in streaks
            if s.threshold == BULL_PEAK
            and (bull_floor.started_bn < s.started_bn < bull_floor.ended_bn)
        ]
        if len(bull_peaks) > 0:
            return 1

    return 0


def get_input_data(
    pair_addr: Hex, block_num: int, hd: HistoricalData
) -> InputData | None:
    hybrid = get_hybrid()
    analysis = hybrid.analyze_pair(pair_addr, block_num, hd)
    if analysis is None:
        return None
    input_data = InputData.from_analysis(analysis)
    if input_data is None:
        return None
    return input_data


def get_scores_dir(model_name: str) -> Path:
    return PROJECT_DIR / "scores" / model_name


def gen_samples_from_pair(pair_addr: Hex, model_name: str) -> Literal[True] | None:
    score_file = get_scores_dir(model_name) / f"{pair_addr}.json.bz2"
    data_file = SAMPLES_DIR / f"{pair_addr}.json.bz2"
    if data_file.exists() and score_file.exists():
        return

    addr_short = str(pair_addr)[:6]
    hybrid = get_hybrid()

    hybrid.sync_blocks(None)

    first_swap_bn = hybrid.get_first_swap_bn(pair_addr)
    if first_swap_bn is None:
        print(f"{addr_short}: skipping, no swaps yet")
        return

    sample_dur = timedelta(hours=HOURS_OF_SAMPLING)
    stop_at = hybrid.get_block_dt(first_swap_bn) + sample_dur

    wait_until = stop_at + timedelta(hours=4)
    now = datetime.now(UTC)
    if now < wait_until:
        hrs_remain = round((wait_until - now).total_seconds() / 3600, 1)
        print(f"{addr_short}: skipping, {hrs_remain} hours until data is available")
        return

    sample_end_bn = hybrid.get_closest_block(stop_at)
    last_needed_bn = sample_end_bn + SCORE_PERIOD_BLKS + 20

    print(f"{addr_short}: fetching historical data...")
    hd = hybrid.get_historical_data(pair_addr, last_needed_bn)

    if len(hd.swaps) == 0:
        raise RuntimeError("no swaps!")

    sample_start_bn = first_swap_bn + SKIP_FIRST_BLOCKS

    assert sample_end_bn > sample_start_bn

    all_block_nums = tuple(range(sample_start_bn, sample_end_bn + 1))

    if not score_file.exists():
        score_df = pl.DataFrame(schema=get_score_polars_schema())
        score_freq: dict[int, int] = {s: 0 for s in range(TOTAL_CLASSES)}
        print(f"{addr_short}: scoring blocks...")
        for block_num in all_block_nums:
            score = get_score(block_num, hd.price_landscape)
            if score is None:
                continue
            score_df.vstack(score_to_df(pair_addr, block_num, score), in_place=True)
            score_freq[score] += 1

        ratio_str = ":".join(str(n) for n in score_freq.values())
        print(f"{addr_short}: scored with ratio {ratio_str}")

        score_file.parent.parent.mkdir(exist_ok=True)
        score_file.parent.mkdir(exist_ok=True)
        score_file.write_bytes(df_to_json(score_df))

    if not data_file.exists():
        data_block_nums = list(all_block_nums)
        random.shuffle(data_block_nums)
        skip_cnt = 0
        data_df = pl.DataFrame(schema=get_data_polars_schema())
        print(f"{addr_short}: building dataset...")
        for block_num in data_block_nums:
            input_data = get_input_data(pair_addr, block_num, hd)
            if input_data is None:
                skip_cnt += 1
                continue
            data_df.vstack(input_data.to_df(), in_place=True)

        done_msg = "done"
        if skip_cnt > 0:
            done_msg += f" (lost {skip_cnt} samples)"
        print(f"{addr_short}: {done_msg}\n")

        data_file.parent.mkdir(exist_ok=True)
        data_file.write_bytes(df_to_json(data_df))
        return True


@define
class SampleInputRow:
    pair_addr: Hex


def collect_samples():
    model_name = input("enter model name: ")

    with open(SAMPLES_CSV) as fl:
        rows = [
            SampleInputRow(
                pair_addr=Hex(row["pair_addr"]),
            )
            for row in csv.DictReader(fl)
        ]

    with alive_bar(
        total=len(rows),
        theme="classic",
        spinner=None,
        enrich_print=False,
        stats="eta: {eta}",
    ) as bar:
        for row in rows:
            addr_short = str(row.pair_addr)[:6]
            try:
                success = gen_samples_from_pair(row.pair_addr, model_name)
                bar(skipped=success is None)
            except Exception as exc:
                print(f"{addr_short}: skipping, {exc}")
                bar(skipped=True)

    print("sampling complete")


def test():
    pair_addr = Hex("0x5023342E6E91C7780042A8389e71632724F7D556")
    gen_samples_from_pair(pair_addr, "test")


if __name__ == "__main__":
    test()
