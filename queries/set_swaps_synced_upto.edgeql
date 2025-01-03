update TokenPair
filter .addr = <str> $pair_addr
set {
    swaps_synced_to_bn := max({.swaps_synced_to_bn, <int32> $block_num})
}