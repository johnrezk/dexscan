CREATE MIGRATION m1omho4p4lxtok3xluzrje2vyqvqxqcpo5bk5dd74y36drcx3hjq4a
    ONTO m1aog5cexuncion2j7ecij3nqq2wer2qgoi74quzeup5fybya4haga
{
  ALTER TYPE default::TokenPair {
      CREATE OPTIONAL PROPERTY swaps_synced_to_bn: std::int32;
  };
};
