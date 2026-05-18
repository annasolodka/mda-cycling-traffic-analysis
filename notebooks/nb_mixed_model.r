library(glmmTMB)
library(readr)
library(dplyr)
library(DHARMa)

train_aggregated <- read_csv("data/processed/train_aggregated.csv")

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
  total_count ~ 
    hour_bin +
    weekday +
    month +
    direction +
    is_public_holiday +
    is_school_holiday +
    fuel_price_petrol_95_rounded +
    offset(log(number_of_periods)) +
    (1 | site_id),
    
  data = train_aggregated,
  
  family = nbinom2(link = "log")
)

summary(nb_mixed_model)
AIC(nb_mixed_model)
exp(fixef(nb_mixed_model)$cond)