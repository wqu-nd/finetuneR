# ==============================================================================
# EXAMPLE: Fine-tuning models with the `finetuneR` package
# ==============================================================================
# This example script demonstrates how to use the package for both a
# classification task and a regression task.

# --- 1. Load Package and Dependencies ---
devtools::install_github("wqu-nd/finetuneR")
library(finetuneR)
library(dplyr)
library(reticulate)
library(purrr)

# --- 2. Setup Python Environment ---
reticulate::use_miniconda("r-reticulate", required = TRUE)
setup_finetuner_env(global_seed = 123)


# ==============================================================================
# PART A: TEXT CLASSIFICATION EXAMPLE
# ==============================================================================

# --- 3A. Load and Prepare Classification Data ---
label_map <- data.frame(
  category = c(
    "Monitoring environmental impact", "Preventing pollution",
    "Strengthening ecosystems", "Reducing use", "Reusing", "Recycling",
    "Repurposing", "Encouraging and supporting others",
    "Educating and training for sustainability",
    "Creating sustainable products and processes",
    "Embracing innovation for sustainability", "Changing how work is done",
    "Choosing responsible alternatives", "Instituting programs and policies",
    "Others"
  ),
  label = 0:14
)

# Create dummy data that mimics the structure of the original dataset
n_samples_class <- 1000
set.seed(11)
my_data <- data.frame(
  text = replicate(n_samples_class, paste(sample(letters, 50, replace = TRUE), collapse = "")),
  category = sample(label_map$category, n_samples_class, replace = TRUE)
) %>%
  left_join(label_map, by = "category")

# --- 4A. Set Model and Training Parameters ---
MODEL_NAME_CLASS <- "distilbert-base-uncased"
NUM_LABELS <- n_distinct(my_data$label)
OUTPUT_DIR_CLASS <- "./finetuneR-classification-results"

# --- 5A. Run Training with Multiple Seeds ---
all_run_results_class <- list()
n_runs <- 5

for (i in 1:n_runs) {
  run_seed <- 11 * i
  cat(paste0("\nExecuting Classification Run ", i, "/", n_runs, " (Seed: ", run_seed, ")..."))

  data_prep_output <- prepare_finetuning_data(
    df = my_data,
    task_type = "classification",
    model_name = MODEL_NAME_CLASS,
    seed = run_seed
  )

  training_args <- create_training_args(
    output_dir = file.path(OUTPUT_DIR, paste0("run_", i)),
    num_train_epochs = 10,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 128,
    learning_rate = 2e-5,
    warmup_steps=0,
    weight_decay=0.05,
    task_type = "classification",
    metric_for_best_model = "precision",
    seed = 11
  )

  all_run_results_class[[paste0("run_", i)]] <- finetune_model(
    datasets = data_preparation_output$datasets,
    task_type = "classification",
    model_name = MODEL_NAME_CLASS,
    training_args = training_args,
    num_labels = NUM_LABELS
  )
}
cat("\n\nAll classification runs complete.\n")

# --- 6A. Generate Classification Report ---
summarize_run_results(
  all_run_results = all_run_results_class,
  task_type = "classification",
  label_map = label_map
)


# ==============================================================================
# PART B: TEXT REGRESSION EXAMPLE
# ==============================================================================

# --- 3B. Load and Prepare Regression Data ---
# Create dummy data for regression (e.g., predicting a sentiment score from 0 to 1)
# In this example, the Pearson's r is expected to be NA,and all the expected value are around 0.5.
# Please your own data for meaningful reason.
n_samples_reg <- 500
set.seed(11)
regression_data <- data.frame(
  text = replicate(n_samples_reg, paste(sample(letters, 40, replace = TRUE), collapse = "")),
  score = runif(n_samples_reg, 0, 1) # Continuous label
)

# --- 4B. Set Model and Training Parameters ---
MODEL_NAME_REG <- "distilbert-base-uncased"
OUTPUT_DIR_REG <- "./finetuneR-regression-results"

# --- 5B. Run a Single Fine-Tuning Run ---
data_prep_output_reg <- prepare_finetuning_data(
  df = regression_data,
  task_type = "regression",
  label_col = "score",
  model_name = MODEL_NAME_REG,
  seed = 11
)

training_args_reg <- create_training_args(
  output_dir = file.path(OUTPUT_DIR_REG, "run_1"),
  num_train_epochs = 4,
  per_device_train_batch_size = 32,
  per_device_eval_batch_size = 64,
  weight_decay = 0.01,
  learning_rate = 3e-5,
  task_type = "regression",
  metric_for_best_model = "mse",
  seed = 11
)

# Store the single run in a list to use the summary function
regression_run_result <- list(
  run_1 = finetune_model(
    datasets = data_prep_output_reg$datasets,
    task_type = "regression",
    model_name = MODEL_NAME_REG,
    training_args = training_args_reg
  )
)

# --- 6B. Generate Regression Report ---
# Because this is a single-run example, the standard deviations are expected to be 0.
# See the previous classification example for the multiple-run setting.
summarize_run_results(
  all_run_results = regression_run_result,
  task_type = "regression"
)

