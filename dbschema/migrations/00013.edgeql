CREATE MIGRATION m1svgkswiuaufv3lhqq3r5jrg25wudv3wms2lildggat6oypry6u3q
    ONTO m1omho4p4lxtok3xluzrje2vyqvqxqcpo5bk5dd74y36drcx3hjq4a
{
  ALTER TYPE default::TokenTransfer {
      CREATE REQUIRED PROPERTY log_index: std::int32 {
          SET REQUIRED USING (<std::int32>1);
      };
      CREATE CONSTRAINT std::exclusive ON ((.block_number, .tx_hash, .log_index));
  };
};
