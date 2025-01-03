with
    block_num := <int32> $block_num
insert Swap {
    tx_hash := <str> $tx_hash,
    pair := (select TokenPair filter .addr = <str> $pair_addr),
    actor_addr := <str> $actor_addr,
    base_amt_change := <bigint> $base_amt_change,
    quote_amt_change := <bigint> $quote_amt_change,
    block_number := block_num
}