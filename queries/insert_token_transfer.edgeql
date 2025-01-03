insert TokenTransfer {
    tx_hash := <str> $tx_hash,
    token := assert_exists((select Token filter .addr = <str> $token_addr)),
    from_addr := <str> $from_addr,
    to_addr := <str> $to_addr,
    amount := <bigint> $amount,
    block_number := <int32> $block_num,
    log_index := <int32> $log_index
}