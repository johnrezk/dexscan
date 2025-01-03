with
  token_addr := <str> $token_addr,
  upto_block := <int32> $block_num,
  token := assert_exists((select Token filter .addr = token_addr)),
  transfers := (
    select TokenTransfer
    filter
      .token = token 
      and .block_number <= upto_block
      and .to_addr != .from_addr
  ),
select (
  (select transfers.to_addr) union (select transfers.from_addr)
)