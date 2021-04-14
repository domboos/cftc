SELECT dat.px_id,
    dat.px_date,
    CASE WHEN dat.px_date < mult.adj_date THEN dat.qty/mult.adj_factor
	ELSE dat.qty END as qty,
	dat2.qty AS oi,
	dat3.qty * mult.multiplier AS px
   FROM cftc.data dat
     LEFT JOIN cftc.cot_desc cd ON dat.px_id = cd.cot_id
	 LEFT JOIN cftc.cot_desc cd2 ON cd.bb_tkr = cd2.bb_tkr and cd.bb_ykey = cd2.bb_ykey
	 LEFT JOIN cftc.fut_desc fut ON cd.bb_tkr = fut.bb_tkr and cd.bb_ykey = fut.bb_ykey
	 LEFT JOIN cftc.fut_mult mult ON cd.bb_tkr = mult.bb_tkr and cd.bb_ykey = mult.bb_ykey
	 LEFT JOIN cftc.data dat2 ON cd2.cot_id = dat2.px_id AND dat.px_date = dat2.px_date
	 LEFT JOIN cftc.data dat3 ON fut.px_id = dat3.px_id AND dat.px_date = dat3.px_date
	 WHERE cd.cot_type::text in ('net_non_commercials'::text, 'net_managed_money'::text)
	 AND cd2.cot_type::text = 'agg_open_interest'::text
	 AND fut.roll::text = 'active_futures'::text AND fut.adjustment::text = 'none'::text) hh
WHERE px is NULL