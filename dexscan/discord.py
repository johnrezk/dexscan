import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from dexscan.database import get_db
from dexscan.hybrid import get_hybrid
from dexscan.primitives import Hex
from dexscan.sampling import InputData
from dexscan.training import get_tf_model


async def send_call_message(
    c: discord.TextChannel, pair_addr: Hex, ticker: str
) -> None:
    embed = discord.Embed(
        title=ticker,
        description=str(pair_addr),
        color=int(Hex(str(pair_addr)[:8])),
        timestamp=datetime.utcnow(),
    )

    view = discord.ui.View()
    view.add_item(
        discord.ui.Button(
            label="Dexscreener",
            style=discord.ButtonStyle.link,
            url=f"https://dexscreener.com/ethereum/{pair_addr}",
        )
    )

    await c.send(
        embed=embed,
        view=view,
    )


def _start_client(tpool: ThreadPoolExecutor):
    load_dotenv()
    discord_token = os.environ["DISCORD_TOKEN"]
    alert_channel_id = int(os.environ["DISCORD_ALERT_CHANNEL_ID"])

    hybrid = get_hybrid()
    db = get_db()
    tf_model = get_tf_model("helios6")

    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix="/", intents=intents)

    @commands.command()
    async def watch(ctx: commands.Context, arg: str):
        try:
            pair_addr = Hex(arg)
            pair = await asyncio.wrap_future(
                tpool.submit(hybrid.get_token_pair, pair_addr)
            )
            base_token = hybrid.get_token(pair.base_token_addr)
            db.set_pair_watching(pair.addr, True)
            await ctx.reply(f"${base_token.ticker} is now being watched")
        except Exception:
            logging.exception("Error in watch command")
            await ctx.reply("An error occured")

    bot.add_command(watch)

    @commands.command("get-channel-id")
    async def get_channel_id(ctx: commands.Context):
        ctx.message.channel.id
        await ctx.reply(f"channel id: {ctx.message.channel.id}")

    bot.add_command(get_channel_id)

    cooldown = timedelta(minutes=20)
    prev_alerts: dict[Hex, datetime] = {}

    @tasks.loop()
    async def bg_loop():
        alert_channel = bot.get_channel(alert_channel_id)
        if not isinstance(alert_channel, discord.TextChannel):
            logging.warn("Alert channel not found")
            await asyncio.sleep(5)
            return
        watched_pairs = hybrid.get_watched_pairs()
        if len(watched_pairs) == 0:
            await asyncio.sleep(5)
            return
        for pair in watched_pairs:
            prev_alert_dt = prev_alerts.get(pair.addr)
            if prev_alert_dt and datetime.utcnow() - prev_alert_dt < cooldown:
                continue
            base_token = hybrid.get_token(pair.base_token_addr)
            try:
                pair_analysis = await asyncio.wrap_future(
                    tpool.submit(hybrid.analyze_pair, pair.addr, None)
                )
            except Exception as exc:
                logging.info(f"stopped watching ${base_token.ticker}: {exc}")
                db.set_pair_watching(pair.addr, False)
                continue
            if pair_analysis is None:
                continue
            input_data = InputData.from_analysis(pair_analysis)
            if input_data is None:
                continue
            if input_data.liq_pool_quote_amt < 3:
                continue
            tf_pred = tf_model.predict(input_data)
            await asyncio.sleep(0)
            if 0.5 < tf_pred.bear or tf_pred.bull < 0.997:
                continue
            await send_call_message(alert_channel, pair.addr, base_token.ticker)
            prev_alerts[pair.addr] = datetime.utcnow()

    async def setup_hook():
        bg_loop.start()

    bot.setup_hook = setup_hook

    bot.run(discord_token)


def start_client():
    with ThreadPoolExecutor(max_workers=3) as tpool:
        _start_client(tpool)
