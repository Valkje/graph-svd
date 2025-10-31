rearrange_raw_files <- function(src_kp_dir, src_acc_dir, overwrite = FALSE) {
  # Transfer files that are non-empty

  pat <- "\\+(3[0-9]+)"

  kp_files <- list.files(src_kp_dir, full.names = TRUE) %>%
    str_subset(pat)
  subjects <- str_match(kp_files, pat)[,2]

  acc_files <- list.files(src_acc_dir, full.names = TRUE) %>%
    str_subset(pat)
  subjects_acc <- str_match(acc_files, pat)[,2]

  if (any(subjects != subjects_acc)) 
    stop("Key press files do not match up with accelerometer files")

  for (i in 1:length(subjects)) {
    sub <- subjects[i]
    kp_path <- kp_files[i]
    acc_path <- acc_files[i]

    sub_dir <- str_glue("sub-{sub}")
    raw_path <- file.path(dat_dir, sub_dir, "raw")
    dir.create(raw_path, showWarnings = FALSE, recursive = TRUE)

    print(str_glue("Processing source key press files for participant {sub}"))
    
    file.copy(
      from = kp_path, 
      to = file.path(raw_path, str_glue("sub-{sub}_keypress.csv")),
      overwrite = overwrite
    )

    print(str_glue("Processing source accelerometer files for participant {sub}"))
    
    file.copy(
      from = acc_path, 
      to = file.path(raw_path, str_glue("sub-{sub}_accelerometer.csv")),
      overwrite = overwrite
    )
  }
}

preproc_biaffect <- function(dat_dir) {
  ### Listing available raw data files
  dirs <- list.dirs(dat_dir)

  pattern <- "sub-([0-9]+)/raw$"
  raw_paths <- str_subset(dirs, pattern)
  subjects <- str_match(raw_paths, pattern)[,2]

  ### Preprocessing for every participant that has both key press and
  ### accelerometer data

  n_sub <- length(subjects)
  dats_acc <- vector("list", n_sub)
  dats_kp <- vector("list", n_sub)
  dats_ses <- vector("list", n_sub)

  for (i in 1:n_sub) {
    tryCatch(
      {
        sub <- subjects[i]

        print(str_glue("Preprocessing participant {sub}"))

        raw_path <- str_subset(raw_paths, str_glue("sub-{sub}"))
        kp_path <- file.path(raw_path, str_glue("sub-{sub}_keypress.csv"))
        acc_path <- file.path(raw_path, str_glue("sub-{sub}_accelerometer.csv"))

        print("Reading data...")
        raw_kp <- read.csv(kp_path)
        raw_acc <- read.csv(acc_path)

        print("Preprocessing accelerometer and key press data...")
        dat_acc <- preproc_acc(raw_acc, sub)

        dats <- preproc_kp(raw_kp, dat_acc, sub)
        dat_kp <- dats$dat_kp
        dat_ses <- dats$dat_ses

        print(str_glue("Saving files for participant {sub}."))

        out_path <- file.path(dat_dir, str_glue("sub-{sub}"), "preproc")
        dir.create(out_path, showWarnings = FALSE, recursive = TRUE)

        out_file <- file.path(out_path, str_glue("sub-{sub}_preprocessed.rda"))
        # save(dat_acc, dat_kp, dat_ses, dat_dist, file = out_file)
        save(dat_acc, dat_kp, dat_ses, file = out_file)

        # Makes it easier to load all participant data in one go
        dats_acc[[i]] <- dat_acc
        dats_kp[[i]] <- dat_kp
        dats_ses[[i]] <- dat_ses
      },
      error = function(e) {
        message(str_glue("Error occurred when processing participant {sub}:"))
        message(e)
        message(str_glue("Skipping participant {sub}."))
      }
    )
  }
  
  # Combine into three large data frames
  dat_acc <- bind_rows(dats_acc)
  dat_kp <- bind_rows(dats_kp)
  dat_ses <- bind_rows(dats_ses)
  
  # Save using high compression
  saveRDS(dat_acc, file = file.path(dat_dir, "dat_acc.rds"), compress = "xz")
  saveRDS(dat_kp, file = file.path(dat_dir, "dat_kp.rds"), compress = "xz")
  saveRDS(dat_ses, file = file.path(dat_dir, "dat_ses.rds"), compress = "xz")
}

# Only runs when called from the command line
if (TRUE || sys.nframe() == 0L) {
  # Set WD to the directory of this file
  setwd("~/Documents/clear3-ica/src")
  
  library(tidyverse)
  
  source("preproc.R")
  
  readRenviron("../.env")
  
  src_dir <- Sys.getenv("SRC_DIR")
  src_kp_dir <- file.path(src_dir, Sys.getenv("SRC_KP_FOLDER"))
  src_acc_dir <- file.path(src_dir, Sys.getenv("SRC_ACC_FOLDER"))
  skip_src_transfer <- Sys.getenv("SKIP_SRC_TRANSFER") == "TRUE"
  
  dat_dir <- Sys.getenv("DAT_DIR")

  dir.create(dat_dir, showWarnings = FALSE, recursive = TRUE)

  if (!skip_src_transfer)
    rearrange_raw_files(src_kp_dir, src_acc_dir)

  preproc_biaffect(dat_dir)
}