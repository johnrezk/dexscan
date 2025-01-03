import sys
from datetime import datetime
from decimal import Decimal
from typing import Callable, Iterable

from rich import box
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from dexscan.hybrid import PairAnalysis, get_hybrid
from dexscan.primitives import Hex, Holder
from dexscan.rpc import get_rpc
from dexscan.sampling import InputData
from dexscan.training import Prediction, get_lgbm_model, get_tf_model

# MAIN


def pc_to_bars(percent: float | Decimal) -> str:
    full = "██"
    thrq = "█▌"
    half = "█"
    oneq = "▌"
    tiny = "▏"
    percent = min(200, percent)
    res = full * int(percent // 10)
    remainder = percent % 10
    if remainder > 8:
        res += full
    elif remainder > 6:
        res += thrq
    elif remainder > 4:
        res += half
    elif remainder > 1:
        res += oneq
    else:
        res += tiny
    res += f" {round(percent, 1)}%"
    return res


def gen_waterlevels(
    holders: Iterable[Holder], spread: tuple[int, int, int] = (50, 100, 200)
):
    water_level1: float = 0
    water_level2: float = 0
    water_level3: float = 0
    water_level4: float = 0
    water_level5: float = 0

    for holder in holders:
        qp = holder.unrealized_quote_percent
        upc = holder.get_unreal_pc()
        if upc > spread[2]:
            water_level1 += qp
        elif upc > spread[1]:
            water_level2 += qp
        elif upc > spread[0]:
            water_level3 += qp
        elif upc > -10:
            water_level4 += qp
        else:
            water_level5 += qp

    return (
        f"[green]{spread[2]:>3}% {pc_to_bars(water_level1)}\n"
        f"[green]{spread[1]:>3}% {pc_to_bars(water_level2)}\n"
        f"[green]{spread[0]:>3}% {pc_to_bars(water_level3)}\n"
        f"[cyan ]   - {pc_to_bars(water_level4)}\n"
        f"[red  ]loss {pc_to_bars(water_level5)}"
    )


def render_view(
    analysis: PairAnalysis,
    to_dec: Callable[[int], Decimal],
    prediction: Prediction,
    tree_pred: float,
):
    holders = analysis.holders

    ap = [h for h in analysis.holders if h.is_active]
    ap.sort(
        key=lambda h: (to_dec(h.unrealized_quote_amt), h.unrealized_pnl),
        reverse=True,
    )

    if len(ap) == 0:
        return "Loading..."

    whale_table = Table(box=box.ROUNDED)
    whale_table.add_column("Addr", no_wrap=True)
    whale_table.add_column("Unrl %", justify="right")
    whale_table.add_column("Unrl PNL", justify="right")
    whale_table.add_column("Unrl Qt", justify="right")
    whale_table.add_column("%", justify="right")
    whale_table.add_column("Activity")
    whale_table.add_column("Traits", justify="right")
    for holder in ap[:29]:
        unreal_pc = holder.get_unreal_pc()
        unreal_pc_text = f"{round(unreal_pc)}%"
        if unreal_pc < -10:
            unreal_pc_text = f"[red]{unreal_pc_text}"
        elif unreal_pc > 200:
            unreal_pc_text = f"[green]{unreal_pc_text}"

        traits = ""
        traits += "🎯" if holder.is_sniper else "⚫"
        traits += "👑" if holder.get_total_pc() > 300 else "⚫"
        traits += "💎" if not holder.has_sold else "🔻"

        whale_table.add_row(
            str(holder.addr),
            unreal_pc_text,
            str(to_dec(holder.unrealized_pnl)),
            str(to_dec(holder.unrealized_quote_amt)),
            str(round(holder.unrealized_quote_percent, 1)),
            holder.last_action,
            traits,
        )

    active_cnt = len(ap)

    pcdp = round(
        100 * len([h for h in ap if h.get_unreal_pc() > 50]) / active_cnt,
        1,
    )
    pcaw = round(
        100 * len([h for h in ap if h.get_unreal_pc() > 0]) / active_cnt,
        1,
    )

    snipers = list(h for h in holders if h.is_sniper)

    stats_txt = (
        f"total unrealized pnl    : {analysis.total_unreal_pnl} eth\n"
        f"total realized pnl      : {analysis.total_real_pnl} eth\n"
        f"total unrealized profit : {analysis.total_unreal_profit} eth\n"
        f"percent deep in profit  : {pcdp}%\n"
        f"percent above water     : {pcaw}%"
    )
    stats_panel = Panel(stats_txt, title="Stats")

    winners = "\n".join(
        f"{w.addr} : {to_dec(w.realized_pnl)} eth"
        for w in sorted(holders, key=lambda h: h.realized_pnl, reverse=True)[:5]
    )
    winners_panel = Panel(winners, title="Winners")

    top_columns = Columns([stats_panel, winners_panel], align="left")

    wl_text = gen_waterlevels(ap)
    water_panel = Panel(wl_text, title="All")

    swl_text = gen_waterlevels(snipers)
    sniper_water_panel = Panel(swl_text, title="Snipers")

    # sniper_text = (
    #     f"spent sniping     : {analysis.snipe_qt_spent_sniping} eth\n"
    #     f"unrealized pnl    : {analysis.snipe_unreal_pnl} eth\n"
    #     f"realized profit   : {analysis.snipe_real_profit} eth\n"
    #     f"{datetime.now()}"
    # )
    # sniper_panel = Panel(sniper_text, title="Snipers")

    ai_texts: list[tuple[float, str]] = [
        (prediction.bull, f"[green ]BULL {pc_to_bars(100 * prediction.bull)}"),
        (prediction.chop, f"[grey66]CHOP {pc_to_bars(100 * prediction.chop)}"),
        (prediction.bear, f"[red1  ]BEAR {pc_to_bars(100 * prediction.bear)}"),
    ]
    ai_texts.sort(key=lambda t: t[0], reverse=True)

    ai_text_final = "\n".join(
        [t[1] for t in ai_texts]
        + [
            f"[white]{datetime.now()}",
            # f"[green]TREE {pc_to_bars(100*tree_pred)}",
        ]
    )
    ai_panel = Panel(ai_text_final, title="AI Prediction")

    mid_columns = Columns([water_panel, sniper_water_panel, ai_panel], align="left")

    layout = Layout()
    layout.split_column(
        Layout(top_columns, name="upper", size=7),
        Layout(mid_columns, name="mid", size=7),
        Layout(whale_table, name="lower"),
    )

    return layout


def main():
    raw_pair_addr = sys.argv[1]
    pair_addr = Hex(raw_pair_addr)

    hybrid = get_hybrid()
    rpc = get_rpc()
    pair = hybrid.get_token_pair(pair_addr)
    quote_token = hybrid.get_token(pair.quote_token_addr)

    q_dec = Decimal(1) / Decimal(10**quote_token.decimals)

    def to_dec(val: int):
        return round(val * q_dec, 2)

    with Status("Loading AI model..."):
        tf_model = get_tf_model("archeos1")
        lgbm_model = get_lgbm_model("kairo5")

    with Live(
        "Loading...",
        screen=True,
        transient=True,
        vertical_overflow="crop",
    ) as live:
        try:
            ai_prediction: Prediction | None = None
            prev_block: int = 0
            while True:
                current_block = rpc.get_latest_block_num()
                if current_block == prev_block:
                    continue
                prev_block = current_block
                analysis = hybrid.analyze_pair(pair.addr, current_block)
                if analysis is None:
                    live.update("No swaps found yet...")
                    continue
                input_data = InputData.from_analysis(analysis)
                new_prediction = (
                    tf_model.predict(input_data)
                    if input_data
                    else Prediction.placeholder()
                )
                if ai_prediction is None:
                    ai_prediction = new_prediction
                else:
                    # ai_prediction = ai_prediction.rolling_avg(new_prediction, 2)
                    ai_prediction = new_prediction
                lgbm_pred = lgbm_model.predict(input_data) if input_data else 0
                live.update(render_view(analysis, to_dec, ai_prediction, lgbm_pred))
        except KeyboardInterrupt:
            return
