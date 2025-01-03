CREATE MIGRATION m1wsb5y4et76hx4jwjyb3x4hesk7ah7fhsqcwwqlmjjfl5257hpinq
    ONTO m1eupt73mv7ymv3qcoqwoidoilnt6aidvvdivrmemsfzjqqak3xcxa
{
  ALTER TYPE default::Swap {
      CREATE INDEX ON (.block_number);
  };
};
