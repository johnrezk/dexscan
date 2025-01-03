CREATE MIGRATION m15hubjqhexp2642vwqfrymumzx6ucb2ok2tuordnnuxmn7fcnblca
    ONTO m1svgkswiuaufv3lhqq3r5jrg25wudv3wms2lildggat6oypry6u3q
{
  ALTER TYPE default::TokenPair {
      CREATE OPTIONAL PROPERTY is_watching: std::bool;
  };
};
