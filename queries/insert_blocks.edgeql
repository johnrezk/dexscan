with
  new_blocks := <array<tuple<int32, datetime>>> $new_blocks
for new_block in array_unpack(new_blocks)
union (
  insert Block {
    number := new_block.0,
    timestamp := new_block.1
  }
)