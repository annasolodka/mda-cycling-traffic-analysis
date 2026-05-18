library(glmmTMB)
library(readr)
library(dplyr)
library(DHARMa)

train_aggregated <- read_csv("../data/processed/train_aggregated.csv")

train_aggregated <- train_aggregated %>%
  mutate(
    hour_bin = factor(hour_bin),
    weekday = factor(weekday),
    month = factor(month),
    direction = factor(direction),
    site_id = factor(site_id),
    is_public_holiday = factor(is_public_holiday),
    is_school_holiday = factor(is_school_holiday)
  )

nb_mixed_model <- glmmTMB(
  total_count ~ hour_bin + weekday + month + direction +
    is_public_holiday + is_school_holiday +
    fuel_price_petrol_95_rounded +
    hour_bin:weekday + hour_bin:direction +
    offset(log(number_of_periods)) +
    (1 | site_id),
  data = train_aggregated,
  family = nbinom2(link = "log")
)

summary(nb_mixed_model)
AIC(nb_mixed_model)
exp(fixef(nb_mixed_model)$cond)

# =========================================================
# Negative Binomial Mixed Model for Cycling Counts
# =========================================================

# -----------------------------
# 1. Install packages (run once)
# -----------------------------
install.packages("glmmTMB")
install.packages("readr")
install.packages("dplyr")
install.packages("performance")
install.packages("DHARMa")

# -----------------------------
# 2. Load libraries
# -----------------------------
library(glmmTMB)
library(readr)
library(dplyr)
library(performance)
library(DHARMa)

# -----------------------------
# 3. Read data
# -----------------------------
train_aggregated <- read_csv(
  "../data/processed/train_aggregated.csv"
)

# -----------------------------
# 4. Inspect data
# -----------------------------
glimpse(train_aggregated)

summary(train_aggregated$total_count)

# -----------------------------
# 5. Convert categorical variables
# -----------------------------
train_aggregated <- train_aggregated %>%
  mutate(
    hour_bin = factor(hour_bin),
    weekday = factor(weekday),
    month = factor(month),
    direction = factor(direction),
    site_id = factor(site_id),
    is_public_holiday = factor(is_public_holiday),
    is_school_holiday = factor(is_school_holiday)
  )

# -----------------------------
# 6. Fit negative binomial mixed model
# -----------------------------
nb_mixed_model <- glmmTMB(
  
  total_count ~
    
    hour_bin +
    weekday +
    month +
    direction +
    is_public_holiday +
    is_school_holiday +
    fuel_price_petrol_95_rounded +
    
    hour_bin:weekday +
    hour_bin:direction +
    
    offset(log(number_of_periods)) +
    
    (1 | site_id),
  
  data = train_aggregated,
  
  family = nbinom2(link = "log")
)

# -----------------------------
# 7. Model summary
# -----------------------------
summary(nb_mixed_model)

# -----------------------------
# 8. AIC
# -----------------------------
AIC(nb_mixed_model)

# -----------------------------
# 9. Incidence Rate Ratios (IRR)
# -----------------------------
exp(fixef(nb_mixed_model)$cond)

# -----------------------------
# 10. Residual diagnostics
# -----------------------------
simulation_output <- simulateResiduals(nb_mixed_model)

plot(simulation_output)

testDispersion(simulation_output)

testZeroInflation(simulation_output)

# -----------------------------
# 11. Predicted values
# -----------------------------
train_aggregated$predicted_counts <- predict(
  nb_mixed_model,
  type = "response"
)

head(train_aggregated)

# -----------------------------
# 12. Save predictions
# -----------------------------
write_csv(
  train_aggregated,
  "../data/processed/train_aggregated_with_predictions.csv"
)

summary(nb_mixed_model)