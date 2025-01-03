import enum
import math
from decimal import Decimal

from attrs import define, field, frozen, validators
from cattrs import Converter
from hexbytes import HexBytes
from uniswap_universal_router_decoder import RouterCodec
from uniswap_universal_router_decoder._decoder import ContractFunction

from dexscan.primitives import Hex


@enum.unique
class UniswapVersion(str, enum.Enum):
    V2 = "V2"


@frozen
class UniswapV2Calculator:
    base_amt: int = field(validator=validators.ge(0))
    quote_amt: int = field(validator=validators.ge(0))
    k: int
    buy_tax: float = field(validator=(validators.ge(0), validators.le(1)))
    sell_tax: float = field(validator=(validators.ge(0), validators.le(1)))

    @classmethod
    def create(
        cls,
        base_amt: int,
        quote_amt: int,
        buy_tax: float = 0,
        sell_tax: float = 0,
    ):
        return UniswapV2Calculator(
            base_amt=base_amt,
            quote_amt=quote_amt,
            k=base_amt * quote_amt,
            buy_tax=buy_tax,
            sell_tax=sell_tax,
        )

    def buy(self, raw_qt_amt: int) -> int:
        if raw_qt_amt < 0:
            raise ValueError("expected positive value")
        if raw_qt_amt == 0:
            return 0

        full_buy_amt = self.base_amt - (self.k / (self.quote_amt + raw_qt_amt))
        taxed_buy_amt = (1 - self.buy_tax) * full_buy_amt
        return max(0, math.floor(taxed_buy_amt))

    def sell(self, raw_base_amt: int) -> int:
        if raw_base_amt < 0:
            raise ValueError("expected positive value")
        if raw_base_amt == 0:
            return 0
        full_sell_amt = self.quote_amt - (self.k / (self.base_amt + raw_base_amt))
        tax_sell_amt = (1 - self.sell_tax) * full_sell_amt
        return max(0, math.floor(tax_sell_amt))

    def get_price(self, raw_qt_amt: int) -> Decimal:
        """
        Returns current quote/base price
        """
        base_amt_recv = self.buy(raw_qt_amt)
        return Decimal(raw_qt_amt) / Decimal(base_amt_recv)


@define
class _HasPath:
    path: tuple[Hex, ...]

    @property
    def from_token_addr(self) -> Hex:
        return self.path[0]

    @property
    def to_token_addr(self) -> Hex:
        return self.path[-1]


@define
class WrapEthCmd:
    recipient: Hex
    amountMin: int


@define
class UnwrapWethCmd:
    recipient: Hex
    amountMin: int


@define
class V2SwapExactInCmd(_HasPath):
    recipient: Hex
    amountIn: int
    amountOutMin: int
    payerIsSender: bool


@define
class V2SwapExactOutCmd(_HasPath):
    recipient: Hex
    amountOut: int
    amountInMax: int
    payerIsSender: bool


UniswapCmd = WrapEthCmd | UnwrapWethCmd | V2SwapExactInCmd | V2SwapExactOutCmd

_codec = RouterCodec()

_cmd_conv = Converter()
_cmd_conv.register_structure_hook(Hex, lambda val, _: Hex(val))


@define
class ExecuteTx:
    cmds: tuple[UniswapCmd, ...]


def _decode_uniswap_exec_tx(tx_input: Hex) -> ExecuteTx:
    binput = HexBytes(bytes(tx_input))
    _, body = _codec.decode.function_input(binput)
    inputs = body.get("inputs")
    assert isinstance(inputs, list)

    commands: list[UniswapCmd] = []
    for inp in inputs:
        assert isinstance(inp, tuple) and len(inp) == 2
        contract_func, params = inp
        assert isinstance(contract_func, ContractFunction)
        assert isinstance(params, dict)
        match contract_func.function_identifier:
            case "WRAP_ETH":
                commands.append(_cmd_conv.structure(params, WrapEthCmd))
            case "UNWRAP_WETH":
                commands.append(_cmd_conv.structure(params, UnwrapWethCmd))
            case "V2_SWAP_EXACT_OUT":
                commands.append(_cmd_conv.structure(params, V2SwapExactOutCmd))
            case "V2_SWAP_EXACT_IN":
                commands.append(_cmd_conv.structure(params, V2SwapExactInCmd))
            case _:
                raise RuntimeError(
                    f"unsupported func id: {contract_func.function_identifier}"
                )

    return ExecuteTx(tuple(commands))


@define
class SwapExactTokensForEthTx(_HasPath):
    amount_in: int
    amount_out_min: int
    to: Hex
    deadline: int


def _decode_swap_exact_tokens_for_eth(tx_input: Hex):
    if not bytes(tx_input).startswith(_SWAP_EXACT_METHOD_ID):
        raise RuntimeError("invalid method id")
    raw_params = bytes(tx_input)[4:]
    byte_cnt = len(raw_params)
    assert byte_cnt % 32 == 0
    params = [
        Hex(raw_params[i * 32 : (i + 1) * 32]) for i in range(round(byte_cnt // 32))
    ]
    assert len(params) >= 7
    return SwapExactTokensForEthTx(
        amount_in=int(params[0]),
        amount_out_min=int(params[1]),
        to=params[3].force_addr(),
        deadline=int(params[4]),
        # index 5 is number of items in arr, ignore
        path=tuple(h.force_addr() for h in params[6:]),
    )


UniswapTx = SwapExactTokensForEthTx | ExecuteTx


_EXECUTE_METHOD_ID = bytes(Hex("0x3593564c"))
_SWAP_EXACT_METHOD_ID = bytes(Hex("0x18cbafe5"))


def decode_uniswap_tx(tx_input: Hex) -> UniswapTx | None:
    method_id = bytes(tx_input)[:4]
    if method_id == _EXECUTE_METHOD_ID:
        return _decode_uniswap_exec_tx(tx_input)
    if method_id == _SWAP_EXACT_METHOD_ID:
        return _decode_swap_exact_tokens_for_eth(tx_input)
    return None
