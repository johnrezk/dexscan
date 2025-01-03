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
)