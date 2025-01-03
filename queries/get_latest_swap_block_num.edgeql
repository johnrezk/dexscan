with
  pair := assert_exists((
    select TokenPair filter .addr = <str> $pair_addr
  )),
  most_recent_swap := (
    select Swap 
    filter .pair = pair
    order by .block_number desc
    limit 1
  )
select assert_exists(
  pair.swaps_synced_to_bn
  ?? most_recent_swap.block_number
  ?? (pair.first_block_number - 10)
)