insert Block {
    number := <int32> $block_num,
    timestamp := <datetime> $timestamp
}
unless conflict on .number