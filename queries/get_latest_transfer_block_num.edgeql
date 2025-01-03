with
    token := assert_exists((
        select Token filter .addr = <str> $token_addr
    )),
    most_recent_transfer := (
      select TokenTransfer
      filter TokenTransfer.token = token
      order by .block_number desc
      limit 1
    )
select assert_exists(
    token.transfers_synced_to_bn
    ?? most_recent_transfer.block_number
    ?? (token.first_block_number - 10)
)