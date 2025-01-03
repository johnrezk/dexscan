with
  token_addr := <str> $token_addr,
  upto_block := <int32> $upto_block,
  token := assert_exists((select Token filter .addr = token_addr)),
  transfers := (
    select TokenTransfer
    filter
      .token = token 
      and .block_number <= upto_block
      and .to_addr != .from_addr
  ),
  holder_addrs := (
    (select transfers.to_addr) union (select transfers.from_addr)
  )
for holder_addr in holder_addrs
union (
  with
    recv_transfers := (
      select transfers filter .to_addr = holder_addr
    ),
    send_transfers := (
      select transfers filter .from_addr = holder_addr
    )
  select {
    holder_addr := holder_addr,
    balance := (
      <bigint> 0 + sum(recv_transfers.amount) - sum(send_transfers.amount)
    )
  }
)