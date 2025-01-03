CREATE MIGRATION m1hoghovyz676jetc2aimjfqdpo7jrqgoxziihczrcbmeeulehwgaq
    ONTO m1kv3ht24cq3ek6pzos7tbihnjeilgav7ftws2umejwn5wnugl2iyq
{
  CREATE FUNCTION default::get_token_balance(token_addr: std::str, holder_addr: std::str, block_num: std::int32) ->  std::bigint USING (WITH
      token := 
          std::assert_exists((SELECT
              default::Token
          FILTER
              (.addr = token_addr)
          ))
      ,
      all_transfers := 
          (SELECT
              default::TokenTransfer
          FILTER
              (((.token = token) AND (.block_number <= block_num)) AND ((.to_addr = holder_addr) OR (.from_addr = holder_addr)))
          )
      ,
      recv_transfers := 
          (SELECT
              all_transfers
          FILTER
              (.to_addr = holder_addr)
          )
      ,
      send_transfers := 
          (SELECT
              all_transfers
          FILTER
              (.from_addr = holder_addr)
          )
  SELECT
      ((<std::bigint>0 + std::sum(recv_transfers.amount)) - std::sum(send_transfers.amount))
  );
};
