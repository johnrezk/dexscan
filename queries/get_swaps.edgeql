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
order by .block_number asc