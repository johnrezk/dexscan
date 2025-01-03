with
  inputs := <array<tuple<str, str, int32>>> $inputs
for input in array_unpack(inputs)
union (
  select {
    holder_addr := input.1,
    balance_amt := get_token_balance(input.0, input.1, input.2)
  }
)