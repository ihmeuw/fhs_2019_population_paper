# Sort draws from the secular trend in a way that make it continuous with the past data

sort_draws <- function(s, draw_data, past_draw, year_forecast_start, year_forecast_end) {
  draw_data_last   <- draw_data[year_id == year_forecast_end]
  draw_data_first  <- draw_data[year_id == year_forecast_start - 1]
  # Subset each of the important data frames
  pred              <- draw_data[scenario == s]
  sub_past          <- past_draw[scenario == s]
  sub_modeled_last  <- draw_data_last[scenario == s]
  sub_modeled_first <- draw_data_first[scenario == s]

  # Get index order of modeled 2085
  rank_modeled_last <- order(sub_modeled_last[,asfr])

  # Impose the last forecast year order of draws on the order of draws in the modeled 2002
  sub_modeled_first[, draw := rank_modeled_last - 1L]

  # Get index order of actual past (incomplete cohort 2002)
  rank_past <- order(sub_past[, value])
  sub_past[, draw := rank_past - 1L]

  # Calculate the shift, which is the past minus the modeled past
  shift <- merge(sub_past[, .(scenario, draw, value)],
                 sub_modeled_first[,.(scenario, draw, asfr)], by = c("scenario", "draw"))
  shift[, shift := value - asfr]

  # Adjust all other cohort predictions accordingly
  pred[, orig_draw := draw]
  pred[, draw := rank_modeled_last - 1L, by = cohort]
  pred <- merge(pred, shift[,.(scenario, draw, shift)], by = c("scenario", "draw"))
  pred[, asfr := asfr + shift]
  pred[, draw := orig_draw]
  pred[, orig_draw := NULL]
  return(pred)
}
