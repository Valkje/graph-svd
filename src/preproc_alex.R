library(gsignal)
library(dplyr)
library(stringr)
library(lubridate)

filter <- dplyr::filter

# Preprocess the accelerometer data
preproc_acc <- function(raw_acc, verbose = FALSE) {
  # Small wrapper around boolean test and print
  vprint <- function(text) if (verbose) print(text)

  vprint("Converting dates and times to objects...")

  dat_acc <- raw_acc %>%
    group_by(session_timestamp) %>%
    mutate(sampleNumber = 1:n()) %>%
    ungroup() %>%
    mutate(
      # Set timezone to UTC.
      session_timestamp = as_datetime(session_timestamp, tz = "UTC"),
      timestamp = as_datetime(timestamp, tz = "UTC"),
      sessionNumber = cumsum(sampleNumber == 1)
    ) %>%
    select(-any_of(c(
      "firstKP_timestamp", "sampleNumber", "diagnostic", "age", "gender")))

  vprint("Filtering...")

  # Discrete Butterworth filter to get rid of noise
  but <- butter(2, w = 0.8, plane = "z", type = "low", output = "Sos")

  dat_acc <- dat_acc %>%
    group_by(sessionNumber) %>%
    mutate(
      # Filter forwards and backwards
      xFiltered = filtfilt(but, x),
      yFiltered = filtfilt(but, y),
      zFiltered = filtfilt(but, z)
    )

  vprint("Marking as active/upright...")

  dat_acc <- dat_acc %>%
    mutate(
      magnitude = sqrt(xFiltered^2 + yFiltered^2 + zFiltered ^2),
      # If magnitude ~ 1, then phone was stationary, i.e. inactive
      activeSample = magnitude < 0.95 | magnitude > 1.05
    ) %>%
    group_by(sessionNumber) %>%
    mutate(
      activeSes = sum(activeSample) / n() > 0.08,
      xMedian = median(xFiltered),
      zMedian = median(zFiltered),
      upright = zMedian < 0.1 & xMedian >= -0.2 & xMedian <= 0.2,
      bed = !(activeSes | upright)
    )

  vprint("Done with accelerometer data!")

  return(dat_acc)
}

# Classify typing dynamics as one- or two-handed
classify_handedness <- function(IKD, distanceFromPrevious) {
  # If we have less than 4 observations, we might not be able to estimate our
  # model properly
  if (sum(!is.na(IKD) & !is.na(distanceFromPrevious)) < 4) {
    return(NA)
  }

  m <- lm(distanceFromPrevious ~ IKD)

  # If slope > 0 and p < 0.05, classify as one-handed.
  # Otherwise, classify as two-handed.

  slope <- coef(m)[2]
  p_val <- summary(m)$coefficients[2, 4]

  if (slope > 0 && p_val < 0.05) {
    return("one-handed")
  }

  return("two-handed")
}

# Preprocess the key press and accelerometer data, combine them
preproc_kp <- function(raw_kp, dat_acc, verbose = FALSE) {
  # Small wrapper around boolean test and print
  vprint <- function(text) if (verbose) print(text)

  vprint("Aggregating accelerometer data...")

  # Aggregate accelerometer data over sessions
  dat_acc_ses <- dat_acc %>%
    group_by(sessionNumber) %>%
    summarize(
      active = activeSes[1],
      upright = upright[1],
      bed = bed[1]
    )

  vprint("Converting key press dates and times to objects...")

  dat_kp <- raw_kp %>%
    arrange(keypress_timestamp) %>% # Enforce chronological order
    group_by(session_timestamp) %>%
    mutate(sampleNumber = 1:n()) %>%
    ungroup() %>%
    mutate(
      sessionNumber = cumsum(sampleNumber == 1),
      utc_offset = as.integer(str_extract(timezone, "([+-]\\d+):\\d+", group = 1)),
      session_timestamp = as_datetime(session_timestamp, tz = "UTC"),
      session_timestamp_local = session_timestamp + hours(utc_offset),
      keypress_timestamp = as_datetime(keypress_timestamp, tz = "UTC"),
      keypress_timestamp_local = keypress_timestamp + hours(utc_offset),
      # Reformat iPhone type for easier look-up in screen point size table
      phoneType = paste(
        "iPhone",
        str_replace(
          str_match(
            phoneInfo,
            "iPhone\\s?(\\w+(?:\\s\\w+|\\+)?(?:\\s\\w+)?)-?,?")[,2],
          "\\+", " Plus"))
    ) %>%
    group_by(sessionNumber) %>%
    mutate(
      IKD = c(NA, diff(keypress_timestamp)),
      previousKeyType = lag(keypress_type),
      handedness = classify_handedness(IKD, distanceFromPrevious)
    ) %>%
    ungroup() %>%
    mutate(handedness = factor(handedness)) %>%
    select(-any_of(c(
      "firstKP_timestamp", "sampleNumber", "diagnostic", "age", "gender")))

  vprint("Converting screen point distance to centimeters...")

  # This file contains a table with iPhone screen point size info
  screen_specs <- read.delim("iPhone_screen_specs.tsv", strip.white = TRUE)

  dat_kp <- dat_kp %>%
    left_join(screen_specs, by = "phoneType") %>%
    mutate(
      distanceFromPrevious = distanceFromPrevious * scaleFactor / PPI * 2.54,
      distanceFromCenter = distanceFromCenter * scaleFactor / PPI * 2.54,
    ) %>%
    select(!c(logicalWidth:release))

  vprint("Aggregating key press data...")

  # Aggregate over sessions, join with accelerometer data
  dat_ses <- dat_kp %>%
    group_by(sessionNumber) %>%
    mutate(
      autocorrectRate = sum(keypress_type == "autocorrection") / n(),
      backspaceRate = sum(keypress_type == "backspace") / n(),
      totalKeyPresses = n()
    ) %>%
    left_join(dat_acc_ses, by = "sessionNumber") %>%
    filter((keypress_type == "alphanum" & previousKeyType == "alphanum") | 
             # For BiAffect 3 data
             (keypress_type == "alphabet" & previousKeyType == "alphabet")) %>%
    summarize(
      phoneType = phoneType[1],
      medianIKD = median(IKD, na.rm = TRUE),
      percent95IKD = quantile(IKD, .95, na.rm = TRUE),
      madIKD = mad(IKD, na.rm = TRUE),
      handedness = handedness[1],
      autocorrectRate = autocorrectRate[1],
      backspaceRate = backspaceRate[1],
      session_timestamp = session_timestamp[1],
      session_timestamp_local = session_timestamp_local[1],
      totalKeyPresses = totalKeyPresses[1],
      active = active[1],
      upright = upright[1],
      bed = bed[1],
      hour = format(session_timestamp, "%H"),
      date = as.Date(session_timestamp),
      timezone = timezone[1],
      utc_offset = utc_offset[1]
    )

  vprint("Done with key press data!")

  # Return non-aggregated and aggregated data
  return(list("dat_kp" = dat_kp, "dat_ses" = dat_ses))
}
