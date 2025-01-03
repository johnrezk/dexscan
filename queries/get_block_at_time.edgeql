select assert_exists((
    select Block {number, timestamp}
    filter .timestamp <= <datetime> $dt
    order by .number desc
    limit 1
))