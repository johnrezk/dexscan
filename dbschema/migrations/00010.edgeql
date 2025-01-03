CREATE MIGRATION m1iu7est5y5fh25eskbug5arvj4cgzgphgd5tdjvyiaexoddtf752a
    ONTO m1tsgnh24y7n4iwduhju4fcscclczdaacqf2ecy2cgdqvqehx4lpya
{
  ALTER FUNCTION default::get_token_balance(token_addr: std::str, holder_addr: std::str, block_num: std::int32) USING (WITH
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
              ((((.token = token) AND (.block_number <= block_num)) AND ((.to_addr = holder_addr) OR (.from_addr = holder_addr))) AND (.to_addr != .from_addr))
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
