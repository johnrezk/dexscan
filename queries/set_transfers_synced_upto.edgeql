update Token 
filter .addr = <str> $token_addr
set {
    transfers_synced_to_bn := max({.transfers_synced_to_bn, <int32> $block_num})
}