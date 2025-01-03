CREATE MIGRATION m1sypxopvusd4yrycphrfjjlhbcsvbfdvl6jg6qubh2nfgy6274ila
    ONTO initial
{
  CREATE TYPE default::Block {
      CREATE REQUIRED PROPERTY number: std::int32 {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE PROPERTY timestamp: std::datetime;
  };
  CREATE SCALAR TYPE default::str_addr EXTENDING std::str {
      CREATE CONSTRAINT std::regexp('^0x[0-9a-f]{40}$');
  };
  CREATE TYPE default::HistoricalBalance {
      CREATE REQUIRED LINK block: default::Block;
      CREATE REQUIRED PROPERTY holder_addr: default::str_addr;
      CREATE REQUIRED PROPERTY token_addr: default::str_addr;
      CREATE CONSTRAINT std::exclusive ON ((.token_addr, .holder_addr, .block));
      CREATE REQUIRED PROPERTY amount: std::bigint;
  };
  CREATE SCALAR TYPE default::str_hash EXTENDING std::str {
      CREATE CONSTRAINT std::regexp('^0x[0-9a-f]{64}$');
  };
  CREATE TYPE default::Token {
      CREATE REQUIRED LINK first_block: default::Block;
      CREATE LINK transfer_sync_upto: default::Block;
      CREATE REQUIRED PROPERTY addr: default::str_addr {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE REQUIRED PROPERTY creator_addr: default::str_addr;
      CREATE REQUIRED PROPERTY decimals: std::int16;
      CREATE REQUIRED PROPERTY first_tx_hash: default::str_hash;
      CREATE REQUIRED PROPERTY name: std::str;
      CREATE REQUIRED PROPERTY ticker: std::str;
  };
  CREATE TYPE default::TokenPair {
      CREATE REQUIRED LINK first_block: default::Block;
      CREATE LINK swap_sync_upto: default::Block;
      CREATE REQUIRED LINK base_token: default::Token;
      CREATE REQUIRED LINK quote_token: default::Token;
      CREATE REQUIRED PROPERTY addr: default::str_addr {
          CREATE CONSTRAINT std::exclusive;
      };
  };
  CREATE TYPE default::Swap {
      CREATE REQUIRED LINK block: default::Block;
      CREATE REQUIRED LINK pair: default::TokenPair;
      CREATE REQUIRED PROPERTY tx_hash: default::str_hash;
      CREATE CONSTRAINT std::exclusive ON ((.tx_hash, .pair));
      CREATE REQUIRED PROPERTY actor_addr: default::str_addr;
      CREATE REQUIRED PROPERTY base_amt_change: std::bigint;
      CREATE REQUIRED PROPERTY quote_amt_change: std::bigint;
  };
  CREATE TYPE default::TokenTransfer {
      CREATE REQUIRED LINK block: default::Block;
      CREATE REQUIRED LINK token: default::Token;
      CREATE REQUIRED PROPERTY to_addr: default::str_addr;
      CREATE INDEX ON ((.token, .to_addr));
      CREATE INDEX ON (.token);
      CREATE REQUIRED PROPERTY from_addr: default::str_addr;
      CREATE INDEX ON ((.token, .from_addr));
      CREATE REQUIRED PROPERTY amount: std::bigint;
      CREATE REQUIRED PROPERTY tx_hash: default::str_hash;
  };
};
