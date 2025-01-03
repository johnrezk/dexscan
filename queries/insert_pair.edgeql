insert TokenPair {
    addr := <str> $pair_addr,
    base_token := (select Token filter .addr = <str> $base_token_addr),
    quote_token := (select Token filter .addr = <str> $quote_token_addr),
    first_block_number := <int32> $first_block_num
}