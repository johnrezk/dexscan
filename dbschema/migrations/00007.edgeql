CREATE MIGRATION m1kv3ht24cq3ek6pzos7tbihnjeilgav7ftws2umejwn5wnugl2iyq
    ONTO m14vhqsaong4v2ocv6ev7lxcl4r27rd2cc4wnjg4o6nnzjxfwzf36q
{
  ALTER TYPE default::Token {
      DROP LINK first_block;
  };
  ALTER TYPE default::TokenPair {
      DROP LINK first_block;
  };
};
