CREATE MIGRATION m1eupt73mv7ymv3qcoqwoidoilnt6aidvvdivrmemsfzjqqak3xcxa
    ONTO m1glxiflgsq5nyjq5lak2iyz4cjbggqi7j4klba6hy3pqrhrysr4cq
{
  ALTER TYPE default::Swap {
      CREATE REQUIRED PROPERTY block_number: std::int32 {
          SET REQUIRED USING (<std::int32>.block.number);
      };
  };
  ALTER TYPE default::TokenTransfer {
      DROP LINK block;
  };
};
