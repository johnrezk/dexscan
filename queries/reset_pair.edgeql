with
  pair := assert_exists((select TokenPair filter .addr = <str> $pair_addr)),
  del_transfers := (
    delete TokenTransfer
    filter .token = pair.base_token
  ),
  del_swaps := (
    delete Swap
    filter .pair = pair
  ),
  upd_pair := (
    update pair
    set { swaps_synced_to_bn := {} }
  ),
  upd_token := (
    update pair.base_token
    set { transfers_synced_to_bn := {} }
  )
select {
  mod_cnt := count(
    del_transfers union del_swaps union upd_pair union upd_token
  )
}