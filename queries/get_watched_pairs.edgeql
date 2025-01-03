select TokenPair {
    addr,
    base_token_addr := .base_token.addr,
    first_block_number
}
filter .is_watching = True