CREATE MIGRATION m14vhqsaong4v2ocv6ev7lxcl4r27rd2cc4wnjg4o6nnzjxfwzf36q
    ONTO m1ypdbhc46jbviozdg5qdphlcxuorr7hxwafimnlrfa2ah4h4qmviq
{
  DROP TYPE default::HistoricalBalance;
  ALTER TYPE default::Token {
      DROP LINK transfer_sync_upto;
  };
  ALTER TYPE default::Token {
      CREATE REQUIRED PROPERTY first_block_number: std::int32 {
          SET REQUIRED USING (<std::int32>.first_block.number);
      };
  };
  ALTER TYPE default::TokenPair {
      DROP LINK swap_sync_upto;
  };
  ALTER TYPE default::TokenPair {
      CREATE REQUIRED PROPERTY first_block_number: std::int32 {
          SET REQUIRED USING (<std::int32>.first_block.number);
      };
  };
};
