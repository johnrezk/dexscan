CREATE MIGRATION m1glxiflgsq5nyjq5lak2iyz4cjbggqi7j4klba6hy3pqrhrysr4cq
    ONTO m1sypxopvusd4yrycphrfjjlhbcsvbfdvl6jg6qubh2nfgy6274ila
{
  ALTER TYPE default::TokenTransfer {
      CREATE REQUIRED PROPERTY block_number: std::int32 {
          SET REQUIRED USING (<std::int32>(SELECT
              .block.number
          ));
      };
      CREATE INDEX ON (.block_number);
  };
};
