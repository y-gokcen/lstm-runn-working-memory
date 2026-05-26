#Assuming we have dataframes containing surprisal values from LSTM/RNN models of different and includes corpus information of Natural Stories (Futrell et al., 2021) and working memory (WM) predictors calculated for each word in the corpus (Shain et al., 2022) training seeds

#Libraries, define WM predictor families (a priori)
library(tidyverse)

DLT_measures <- c(
  "DLT", "DLT_S", "DLT_C", "DLT_V",
  "DLT_M", "DLT_VM", "DLT_CM", "DLT_CV", "DLT_CVM"
)

LC_measures <- c(
  "Start_Embd", "Embd_Depth",
  "End_MCE_DRV", "End_MCE_DR", "End_MCE_WD",
  "End_CE",
  "Const_Len_DRV", "Const_Len_DR", "Const_Len_WD",
  "End_of_Const"
)

WM_predictors <- c(DLT_measures, LC_measures, "ACT-R")

#Compute resiudals between LSTM and RNNs
compute_residuals_LSTM_RNN <- function(df) {

  lstm_cols <- grep("^LSTM\\d+", names(df), value = TRUE)
  relu_cols <- grep("^RELU\\d+", names(df), value = TRUE)
  tanh_cols <- grep("^TANH\\d+", names(df), value = TRUE)

  message(
    "Found ", length(lstm_cols), " LSTM, ",
    length(relu_cols), " RELU, ",
    length(tanh_cols), " TANH columns."
  )

  safe_resid <- function(y, x) {
    ok <- complete.cases(y, x)
    fit <- lm(y[ok] ~ x[ok])
    res <- rep(NA_real_, length(y))
    res[ok] <- resid(fit)
    res
  }

  ## LSTM–RELU residuals
  for (lstm_col in lstm_cols) {
    lstm_num <- gsub("LSTM", "", lstm_col)
    for (relu_col in relu_cols) {
      relu_num <- gsub("RELU", "", relu_col)
      resid_name <- paste0("resLR_", lstm_num, "_", relu_num)
      df[[resid_name]] <- safe_resid(df[[lstm_col]], df[[relu_col]])
    }
  }

  ## LSTM–TANH residuals
  for (lstm_col in lstm_cols) {
    lstm_num <- gsub("LSTM", "", lstm_col)
    for (tanh_col in tanh_cols) {
      tanh_num <- gsub("TANH", "", tanh_col)
      resid_name <- paste0("resLT_", lstm_num, "_", tanh_num)
      df[[resid_name]] <- safe_resid(df[[lstm_col]], df[[tanh_col]])
    }
  }

  df
}

#Build WM x Model coefficient table
build_wm_df <- function(df, WM_predictors) {

  resid_cols <- grep("^resL[RT]_", names(df), value = TRUE)

  results <- vector("list", length(resid_cols) * length(WM_predictors))
  k <- 1

  for (resid in resid_cols) {
    for (wm in WM_predictors) {

      fml <- as.formula(paste(wm, "~", resid))
      fit <- try(lm(fml, data = df), silent = TRUE)
      if (inherits(fit, "try-error")) next

      sm <- summary(fit)
      if (!(resid %in% rownames(sm$coefficients))) next

      results[[k]] <- data.frame(
        WM_Predictor = wm,
        ID           = resid,
        coef         = sm$coefficients[resid, "Estimate"],
        p_value      = sm$coefficients[resid, "Pr(>|t|)"],
        stringsAsFactors = FALSE
      )

      k <- k + 1
    }
  }

  bind_rows(results)
}

#Run pipeline
all_iter_models_raw <- compute_residuals_LSTM_RNN(all_iter_models_raw)

wm_df <- build_wm_df(
  df = all_iter_models_raw,
  WM_predictors = WM_predictors
)

#Apply Bonferroni correction within families
wm_df <- wm_df %>%
  mutate(
    WM_family = case_when(
      WM_Predictor %in% DLT_measures ~ "DLT",
      WM_Predictor %in% LC_measures  ~ "LC"
    )
  ) %>%
  group_by(WM_family) %>%
  mutate(
    p_bonf = p.adjust(p_value, method = "bonferroni"),
    sigs   = ifelse(p_bonf < 0.05, "sig", "not_sig")
  ) %>%
  ungroup()

#Clean model IDs for plotting
wm_df <- wm_df %>%
  mutate(
    Model = case_when(
      grepl("^resLR_", ID) ~ gsub("resLR_(\\d+)_(\\d+)", "LR\\1.\\2", ID),
      grepl("^resLT_", ID) ~ gsub("resLT_(\\d+)_(\\d+)", "LT\\1.\\2", ID)
    )
  )

#Plot raster with outline significance
df <- wm_df %>%
  rename(
    coef = `Estimated Coefficient`,
    predictor = WM_Predictor,
    model = ID
  ) %>%
  mutate(
    predictor = factor(predictor, levels = rev(unique(predictor))),
    model = factor(model, levels = unique(model)),
    sig_flag = sigs == "sig"
  )

p <- ggplot(df, aes(x = model, y = predictor, fill = coef)) +
  geom_tile(color = NA) +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limits = c(-0.4, 0.3),
    oob = scales::squish,
    name = "Estimated Coefficient"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    panel.grid = element_blank()
  ) +
  labs(x = NULL, y = "WM Demand Measure")

p +
  geom_tile(
    data = df %>% filter(sig_flag),
    aes(x = model, y = predictor),
    fill = NA,
    color = "black",
    linewidth = 0.3
  )

p +
  geom_tile(
    data = df %>% filter(sig_flag),
    aes(x = model, y = predictor),
    fill = NA,
    color = "black",
    linewidth = 0.3
  ) +
  theme(
    axis.text.y = element_text(size = 11),
    legend.position = "bottom",
    legend.key.width = unit(2, "cm")
  )
