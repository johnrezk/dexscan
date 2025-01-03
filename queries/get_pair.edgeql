select TokenPair {
    id,
    addr,
    base_token_addr := .base_token.addr,
    quote_token_addr := .quote_token.addr,
    first_block_num := .first_block_number
} filter .addr = <str> $pair_addr