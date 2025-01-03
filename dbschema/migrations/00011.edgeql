CREATE MIGRATION m1aog5cexuncion2j7ecij3nqq2wer2qgoi74quzeup5fybya4haga
    ONTO m1iu7est5y5fh25eskbug5arvj4cgzgphgd5tdjvyiaexoddtf752a
{
  ALTER TYPE default::Token {
      CREATE OPTIONAL PROPERTY transfers_synced_to_bn: std::int32;
  };
};
