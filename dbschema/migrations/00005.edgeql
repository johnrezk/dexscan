CREATE MIGRATION m1ypdbhc46jbviozdg5qdphlcxuorr7hxwafimnlrfa2ah4h4qmviq
    ONTO m1wsb5y4et76hx4jwjyb3x4hesk7ah7fhsqcwwqlmjjfl5257hpinq
{
  ALTER TYPE default::Block {
      ALTER PROPERTY timestamp {
          SET REQUIRED USING (<std::datetime>std::assert_exists(.timestamp));
      };
  };
  ALTER TYPE default::Swap {
      DROP LINK block;
  };
};
