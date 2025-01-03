import csv
from collections import defaultdict
from datetime import datetime
from typing import Callable

import cattrs
import orjson
from attrs import define, field, frozen
from ciso8601 import parse_datetime
from rich.progress import track
from rich.status import Status

from dexscan.constants import PROJECT_DIR
from dexscan.hybrid import get_hybrid
from dexscan.primitives import Hex
from dexscan.rpc import get_rpc
from dexscan.sampling import InputData
from dexscan.training import KerasModelWrapper, Prediction, get_tf_model
from dexscan.utils import percent_change

GAS_COST_ETH = round(0.007 * (10**18))
HAND_SIZE_ETH = round(0.1 * (10**18))
TRADING_BLOCKS = 5 * 60 * 3


conv = cattrs.Converter()
conv.register_structure_hook(datetime, lambda v, _: parse_datetime(v))
conv.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
conv.register_structure_hook(Hex, lambda v, _: Hex(v))
conv.register_unstructure_hook(Hex, lambda h: str(h))


@frozen
class Trade:
    pair_addr: Hex
    buy_block: int
    buy_dt: datetime
    # buy_pred: Prediction
    buy_ath_pc_from: float
    buy_ath_mins_since: float
    buy_mrh_pc_from: float
    buy_mrh_mins_since: float
    sell_block: int
    sell_dt: datetime
    # sell_pred: Prediction | None
    pnl: float
    hit_target: bool


@frozen
class TestResult:
    eth_balance: int
    trades_taken: int


TestFunc = Callable[[InputData], bool]


@define
class ModelState:
    eth_balance: int = field(init=False, default=0)
    token_balance: int = field(init=False, default=0)
    buy_block: int | None = field(init=False, default=None)
    buy_data: InputData | None = field(init=False, default=None)
    buy_pred: Prediction | None = field(init=False, default=None)
    buy_cooldown: int = field(init=False, default=0)
    trades: list[Trade] = field(init=False, factory=list)


def run_one_test(
    pair_addr: Hex, models: dict[str, KerasModelWrapper]
) -> dict[str, list[Trade]] | None:
    rpc = get_rpc()
    latest_block = rpc.get_latest_block_num()

    hybrid = get_hybrid()

    hybrid.sync_blocks(None)

    first_block = hybrid.get_first_swap_bn(pair_addr)
    if first_block is None:
        print("no swaps, skipping")
        return None

    test_start = first_block + 5 * 30
    test_end = test_start + 5 * 60 * 5
    force_end = test_end + TRADING_BLOCKS

    if latest_block < force_end:
        print("not enough data to test, skipping")
        return None

    with Status("getting historical data..."):
        hd = hybrid.get_historical_data(pair_addr, force_end)

    state_per_model = {model_name: ModelState() for model_name in models}

    for block_num in track(
        range(test_start, force_end + 1), "running...", transient=True
    ):
        for model_name, s in state_per_model.items():
            model = models[model_name]
            if s.buy_block is None:
                if block_num > test_end:
                    break
                if s.buy_cooldown > 0:
                    s.buy_cooldown -= 1
                    continue
                analysis = hybrid.analyze_pair(pair_addr, block_num, hd)
                if analysis is None:
                    continue
                data = InputData.from_analysis(analysis)
                if data is None:
                    continue
                prediction = model.predict(data)
                if prediction.bull < 0.98:
                    continue
                s.buy_block = block_num + 3
                s.token_balance += hd.price_landscape.buy(s.buy_block, HAND_SIZE_ETH)
                s.eth_balance -= HAND_SIZE_ETH + GAS_COST_ETH
                s.buy_data = data
                s.buy_pred = prediction
            else:
                if block_num <= s.buy_block:
                    continue
                assert s.buy_pred and s.buy_data
                blocks_since_buy = block_num - s.buy_block
                eth_value = hd.price_landscape.sell(block_num, s.token_balance)
                pc_change = percent_change(HAND_SIZE_ETH, eth_value)
                hit_target = 35 < pc_change
                # sell_pred: Prediction | None = None
                # analysis = hybrid.analyze_pair(pair_addr, block_num, hd)
                # if not analysis:
                #     continue
                # data = InputData.from_analysis(analysis)
                # if not data:
                #     continue
                # sell_pred = model.predict(data)
                if (
                    hit_target  # (hit_target and 0.7 < sell_pred.bear)
                    or pc_change < -20
                    or TRADING_BLOCKS < blocks_since_buy
                    or block_num == force_end
                ):
                    sell_block = block_num + 3
                    true_value = hd.price_landscape.sell(sell_block, s.token_balance)
                    pnl = round(
                        number=(true_value - (HAND_SIZE_ETH + 2 * GAS_COST_ETH))
                        / (10**18),
                        ndigits=4,
                    )
                    s.trades.append(
                        Trade(
                            pair_addr=pair_addr,
                            buy_block=s.buy_block,
                            buy_dt=hybrid.get_block_dt(s.buy_block),
                            # buy_pred=buy_pred,
                            buy_ath_pc_from=s.buy_data.ath_pc_from,
                            buy_ath_mins_since=s.buy_data.ath_mins_since,
                            buy_mrh_pc_from=s.buy_data.mrh_pc_from,
                            buy_mrh_mins_since=s.buy_data.mrh_mins_since,
                            sell_block=sell_block,
                            sell_dt=hybrid.get_block_dt(sell_block),
                            # sell_pred=sell_pred,
                            pnl=pnl,
                            hit_target=hit_target,
                        )
                    )
                    s.eth_balance += true_value - GAS_COST_ETH
                    s.token_balance = 0
                    s.buy_block = None
                    s.buy_cooldown = 4
                    s.buy_pred = None
                    s.buy_data = None

    return {model_name: s.trades for model_name, s in state_per_model.items()}


def run_backtest():
    backtest_name = "bt1"
    model_names = ["caper4"]

    samples_file = PROJECT_DIR / "samples_test.csv"
    with open(samples_file) as fl:
        reader = csv.DictReader(fl)
        pair_addrs = [Hex(row["pair_addr"]) for row in reader]

    models = {model_name: get_tf_model(model_name) for model_name in model_names}

    trades_per_model: dict[str, list[Trade]] = defaultdict(list)
    for pair_addr in pair_addrs:
        print(f"testing {pair_addr}")
        trades_res = run_one_test(pair_addr, models)
        if trades_res is None:
            continue
        for model_name, trades in trades_res.items():
            trades_per_model[model_name].extend(trades)
            current_balance = round(
                sum(trade.pnl for trade in trades_per_model[model_name]), 3
            )
            print(f"{model_name.rjust(8)}: {current_balance} eth")

    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    for model_name, all_trades in trades_per_model.items():
        results_file = results_dir / f"{backtest_name}-{model_name}.json"
        results_file.write_bytes(orjson.dumps(conv.unstructure(all_trades)))


@frozen
class TradeAbridged:
    pnl: float
    hit_target: bool


def analyze_results():
    bt_name = "bt1"
    results_dir = PROJECT_DIR / "results"
    bt_files = list(results_dir.glob(f"{bt_name}-*.json"))
    if len(bt_files) == 0:
        print("no results found")
        return
    for bt_file in bt_files:
        model_name = bt_file.stem.split("-")[1]
        trades = conv.structure(orjson.loads(bt_file.read_bytes()), list[TradeAbridged])
        success_trades = [trade for trade in trades if trade.hit_target]
        final_balance = round(sum(trade.pnl for trade in trades), 3)

        trade_cnt = len(trades)
        pc_success = round(100 * len(success_trades) / len(trades))
        print(
            f"{model_name.rjust(6)}:  {pc_success}% {str(trade_cnt).rjust(5)} trades   {final_balance/trade_cnt:.3f} eth/ea"
        )


if __name__ == "__main__":
    run_backtest()
