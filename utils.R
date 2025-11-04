library(r2mlm)
library(ggplot2)
library(dplyr)

plot_ev <- function(model, bargraph = FALSE) {
  cols <- rev(c("skyblue2", "skyblue3", "skyblue4", "grey20"))
  
  r_squared <- r2mlm(model, bargraph = bargraph)
  
  vars <- r_squared$Decompositions[,1]
  vars_df <- data.frame(variance = unclass(vars)) %>%
    rownames_to_column("component") %>%
    filter(component != "fixed, between") %>%
    mutate(
      component = case_when(
        component == "fixed, within" | component == "fixed" ~ "Fixed effects",
        component == "slope variation" ~ "Random slope",
        component == "mean variation" ~ "Random intercept",
        component == "sigma2" ~ "Residuals"
      )
    )
  
  g_var <- ggplot(vars_df, aes(1, variance)) +
    geom_col(aes(fill = fct_rev(component))) +
    scale_fill_manual(values = cols, name = "Variance component") +
    labs(x = NULL, y = "Proportion of variance explained") +
    scale_x_continuous(breaks = NULL) +
    scale_y_continuous(position = "right") +
    theme_minimal() +
    theme(text = element_text(size = 26))
  
  g_var_hor <- ggplot(vars_df, aes(variance, 1)) +
    geom_col(aes(fill = fct_rev(component)), orientation = "y") +
    scale_fill_manual(values = cols, name = "Variance component", 
                      guide = guide_legend(reverse = TRUE)) +
    labs(x = "Proportion of variance", y = NULL) +
    scale_x_continuous(position = "top") +
    scale_y_continuous(breaks = NULL) +
    theme_minimal() +
    theme(
      text = element_text(size = 20),
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 14),
      legend.position = "top"
    )
  
  list(
    vars_df = vars_df,
    g_var = g_var,
    g_var_hor = g_var_hor
  )
}

plot_regression <- function(
    model, 
    dat, 
    predictor, 
    unscaled_pred, 
    response,
    xlab,
    ylab
) {
  hour_mean <- pull(dat, {{predictor}}) %@% "scaled:center"
  hour_sd <- pull(dat, {{predictor}}) %@% "scaled:scale"
  
  hour_sd <- if (is.null(hour_sd)) 1 else hour_sd
  
  pr <- predict_response(model, terms = as_name(ensym(predictor))) %>%
    as.data.frame() %>%
    mutate(x = hour_sd * x + hour_mean)
  
  max_est <- pull(dat, {{unscaled_pred}}) %>% max()
  
  ggplot(pr, aes(x, predicted)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
    geom_line(color = "lightcoral", lwd = 1.2) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), 
                fill="lightcoral", alpha = 0.2) +
    geom_jitter(
      aes({{unscaled_pred}}, {{response}}),
      width = 0.2, height = 0.1,# alpha = 0.5,
      data = dat) +
    scale_x_continuous(minor_breaks = seq(0, max_est)) +
    scale_y_continuous(minor_breaks = seq(0, 15), limits = c(0, 13)) + 
    labs(x = xlab, 
         y = ylab) +
    coord_fixed() +
    theme_minimal() +
    theme(text = element_text(size = 26))
}

plot_reg_ev <- function(
    model, 
    dat, 
    predictor, 
    unscaled_pred, 
    response,
    xlab,
    ylab
) {
  g_pred <- plot_regression(
    model, dat, {{predictor}}, {{unscaled_pred}}, {{response}}, xlab, ylab
  )
  
  ls <- plot_ev(model, bargraph = FALSE)
  
  g_var <- ls$g_var
  g_var_hor <- ls$g_var_hor
  
  ggarrange(g_pred, g_var, widths = c(7, 1), legend.grob = get_legend(g_var_hor))
}

format_tab1 <- function(tab1) {
  format_cont <- function(tab1) {
    tab <- tab1$ContTable$Overall
    
    data.frame(
      variable = c("N", "Age"), 
      mean = c(tab[1, 1], tab[1, 4]), 
      sd = c(NA, tab[1, 5]),
      type = "numeric",
      order = c(1, 2)
    ) %>%
      mutate(
        `__var` = variable,
        variable = case_when(
          is.na(sd) ~ variable,
          TRUE ~ str_glue("Mean {tolower(variable)} (SD)")
        ),
        text = case_when(
          is.na(sd) ~ as.character(mean),
          TRUE ~ str_glue("{round(mean, 2)} ({round(sd, 2)})")
        )
      ) %>%
      select(!c(mean, sd))
  }
  
  format_cat <- function(tab1) {
    format_tab_text <- function(freq, percent, type) {
      texts <- str_glue("{freq} ({round(percent, 2)})")
      
      if (type[1] == "factor")
        texts <- c("", texts)
      
      texts
    }
    
    format_tab_var <- function(variable, level, type) {
      if (type[1] == "factor")
        return(c(paste0(variable[1], " (%)"), paste0("   ", level)))
      
      variable
    }
    
    dict <- c(
      Agoraphobia = "Agoraphobia",
      ADHD = "Attention-Deficit/Hyperactivity Disorder",
      BED = "Binge Eating Disorder",
      BPD = "Borderline Personality Disorder",
      GAD = "Generalised Anxiety Disorder",
      MDD = "Major Depressive Disorder",
      OCD = "Obsessive Compulsive Disorder",
      OtherAnx = "Other Specified Anxiety Disorder",
      Othertrauma = "Other Specified Trauma- and Stressor-Related Disorder",
      PDD = "Persistent Depressive Disorder",
      PTSD = "Posttraumatic Stress Disorder",
      PanicDO = "Panic Disorder",
      Phobi = "Specific Phobia",
      SAD = "Social Anxiety Disorder",
      SUDalc = "Alcohol Use Disorder",
      SUDmj = "Cannabis",
      SUDsha = "Sedative-Hypnotic-Anxiolytic",
      SUDother = "Other/Unknown Substance Use Disorder",
      othDD = "Other Specified Depressive Disorder",
      bipolar2 = "Bipolar II Disorder",
      otheat = "Other Specified Feeding or Eating Disorder",
      otherOC = "Other Specified Obsessive Compulsive and Related Disorder"
    )
    
    lapply(tab1$CatTable$Overall, function(df) {
      df %>%
        select(level, freq, percent) %>%
        filter(level != "FALSE")
    }) %>%
      bind_rows(.id = "variable") %>%
      mutate(
        type = case_when(
          level == "TRUE" ~ "logical",
          TRUE ~ "factor"
        ),
        variable = str_replace(variable, "currentdx_", "")
      ) %>%
      group_by(variable) %>%
      reframe(
        `__var` = variable[1],
        variable = format_tab_var(variable, level, type),
        text = format_tab_text(freq, percent, type),
        type = type[1],
        order = 3
      ) %>%
      mutate(variable = case_when(
        variable %in% names(dict) ~ paste0("   ", dict[variable]),
        TRUE ~ variable
      )) %>%
      arrange(type) %>%
      filter(variable != "PMDD_prov")
  }
  
  bind_rows(
    format_cont(tab1),
    format_cat(tab1)
  )
}

create_tab1 <- function(dem, ids) {
  dem %>%
    filter(id %in% ids) %>%
    select(!c(id, race_clear123, ethnicity, bmi, filing_status, sex_orient)) %>%
    select(where(~ !is.numeric(.x) || sum(.x, na.rm = TRUE) > 0)) %>%
    mutate(across(starts_with("currentdx"), as.logical)) %>%
    CreateTableOne(data = .) %>%
    format_tab1()
}

rmse <- function(m) {
  sqrt(mean(resid(m)^2))
}

mae <- function(m) {
  mean(abs(resid(m)))
}

get_raw_resids <- function(m, response, predictor) {
  m_frame <- model.frame(m)
  
  pred_str <- as_string(ensym(predictor))
  coef_tab <- coef(summary(m))
  
  if (!(pred_str %in% rownames(coef_tab)))
    abort(str_glue("{pred_str} cannot be found in the model coefficient table"))
  
  intercept <- coef_tab["(Intercept)", "Estimate"]
  beta <- coef_tab[pred_str, "Estimate"]
  
  m_frame %>%
    mutate(diff = {{response}} - intercept - beta * {{predictor}}) %>%
    pull(diff)
}

rmse_fixed <- function(m, response, predictor) {
  raw_resids <- get_raw_resids(m, {{response}}, {{predictor}})
  
  sqrt(mean(raw_resids^2))
}

mae_fixed <- function(m, response, predictor) {
  raw_resids <- get_raw_resids(m, {{response}}, {{predictor}})
  
  mean(abs(raw_resids))
}

check_model <- function(m, response, predictor) {
  print(plot(m))
  
  qqnorm(resid(m), main = "Q-Q plot residuals")
  qqline(resid(m))
  
  hist(resid(m), breaks = 30, main = "Histogram residuals", xlab = "Residual")
  
  # Leave subject hardcoded for now
  re <- ranef(m)$subject %>%
    rownames_to_column("subject") %>%
    pivot_longer(!subject)
  
  g <- ggplot(re, aes(sample = value)) +
    geom_qq() +
    geom_qq_line() +
    facet_wrap(~ name, scales = "free") +
    labs(title = "Q-Q plots random effects")
  print(g)
  
  cat("Variance decomposition:\n")
  print(r2mlm(m))
  
  cat("RMSE: ", rmse(m), "\n", sep = "")
  cat("MAE: ", mae(m), "\n", sep = "")
  
  cat("Fixed-effect RMSE: ", rmse_fixed(m, {{response}}, {{predictor}}), 
      "\n", sep = "")
  cat("Fixed-effect MAE: ", mae_fixed(m, {{response}}, {{predictor}}), 
      "\n", sep = "")
}

welch_tests <- function(phase_df) {
  phase_df %>%
    group_by(days_since_start, diff_sign) %>%
    summarize(phases = list(phase)) %>%
    pivot_wider(names_from = diff_sign, values_from = phases) %>%
    mutate(
      welch_test = list(t.test(unlist(Westwards), unlist(Eastwards))),
      df = welch_test[[1]]$parameter,
      stat = welch_test[[1]]$statistic,
      p_value = welch_test[[1]]$p.value,
      stars = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ ""
      )
    )
}

format_welch <- function(df) {
  df %>%
    mutate(
      n_east = sapply(Eastwards, length),
      n_west = sapply(Westwards, length),
      across(df:stat, ~ round(.x, digits = 2)),
      p_value = case_when(
        p_value < 0.0001 ~ "<0.0001",
        TRUE ~ sprintf("%.2g", p_value)
      )
    ) %>%
    select(!c(Westwards, Eastwards, welch_test, stars)) %>%
    relocate(c(n_east, n_west), .before = df)
}
