with
  token := assert_exists((
    select Token filter .addr = <str> $token_addr
  )),
  block_num := <int32> $block_num
select TokenTransfer {**}
filter
  .token = token
  and .block_number <= block_num
  and .to_addr != .from_addr
order by .block_number asc