select Token {
    id,
    addr,
    ticker,
    name,
    decimals,
    creator_addr,
    first_tx_hash,
    first_block_num := .first_block_number
} filter .addr = <str> $token_addr;