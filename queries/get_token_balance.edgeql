with
    token_addr := <str> $token_addr,
    holder_addr := <str> $target_addr,
    block_num := <int32> $upto_block_num
select get_token_balance(token_addr, holder_addr, block_num)