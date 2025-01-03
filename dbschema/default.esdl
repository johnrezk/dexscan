module default {
    scalar type str_addr extending str {
        constraint regexp(r'^0x[0-9a-f]{40}$');
    }

    scalar type str_hash extending str {
        constraint regexp(r'^0x[0-9a-f]{64}$');
    }

    type Token {
        required addr: str_addr {
            constraint exclusive;
        };
        required ticker: str;
        required name: str;
        required decimals: int16;
        required first_block_number: int32;
        required first_tx_hash: str_hash;
        required creator_addr: str_addr;
        optional transfers_synced_to_bn: int32;
    }

    type TokenPair {
        required addr: str_addr {
            constraint exclusive;
        };
        required base_token: Token;
        required quote_token: Token;
        required first_block_number: int32;
        optional swaps_synced_to_bn: int32;
        optional is_watching: bool;
    }

    type TokenTransfer {
        required block_number: int32;
        required tx_hash: str_hash;
        required log_index: int32;

        required token: Token; 
        required from_addr: str_addr;
        required to_addr: str_addr;
        required amount: bigint;

        index on (.token);
        index on (.block_number);
        index on ((.token, .from_addr));
        index on ((.token, .to_addr));

        constraint exclusive on ((.block_number, .tx_hash, .log_index));
    }

    type Swap {
        required tx_hash: str_hash;
        required pair: TokenPair;
        required actor_addr: str_addr;
        required base_amt_change: bigint;
        required quote_amt_change: bigint;
        required block_number: int32;

        index on (.block_number);

        constraint exclusive on ((.tx_hash, .pair));
    }

    type Block {
        required number: int32 {
            constraint exclusive;
        };
        required timestamp: datetime;
    }

    function get_token_balance(token_addr: str, holder_addr: str, block_num: int32) -> bigint
    using (
        with
            token := assert_exists((select Token filter .addr = token_addr)),
            all_transfers := (
                select TokenTransfer
                filter
                    .token = token
                    and .block_number <= block_num
                    and (.to_addr = holder_addr or .from_addr = holder_addr)
                    and (.to_addr != .from_addr)
            ),
            recv_transfers := (
                select all_transfers filter .to_addr = holder_addr
            ),
            send_transfers := (
                select all_transfers filter .from_addr = holder_addr
            )
        select (
            <bigint> 0 + sum(recv_transfers.amount) - sum(send_transfers.amount)
        )
    );
}
