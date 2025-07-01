# ==============================================================================
# EXAMPLE: Fine-tuning a classifier with the `finetuneR` package
# ==============================================================================
# This example uses the original 15-label classification task and demonstrates
# the simplified workflow using the new `summarize_run_results` function.

# --- 1. Load Package and Dependencies ---
#devtools::install_github("wqu-nd/finetuneR")
library(finetuneR)
library(dplyr)
library(reticulate)
library(purrr)

# --- 2. Setup Python Environment ---
reticulate::use_miniconda("r-reticulate", required = TRUE)

#setup
setup_finetuner_env(global_seed = 123)


# --- 3. Load and Prepare Data ---
# Use the original 15-label mapping
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
n_samples <- 1000
set.seed(11) # R-level seed for creating the data
my_data <- data.frame(
  text = replicate(n_samples, paste(sample(letters, 50, replace = TRUE), collapse = "")),
  category = sample(label_map$category, n_samples, replace = TRUE)
) %>%
  left_join(label_map, by = "category")

cat("--- Data Preview ---\n")
print(head(my_data))
cat("\nNumber of labels:", n_distinct(my_data$label), "\n")


# --- 4. Set Model and Training Parameters ---
MODEL_NAME <- "distilbert-base-uncased"
NUM_LABELS <- n_distinct(my_data$label)
OUTPUT_DIR <- "./finetuneR-classification-results"


# --- 5. Run Training with Multiple Seeds ---
all_run_results <- list()
n_runs <- 3 # Use 3 runs for a quicker example

for (i in 1:n_runs) {
  run_seed <- 11 * i
  cat(paste0("\nExecuting Run ", i, "/", n_runs, " (Seed: ", run_seed, ")..."))

  # a. Prepare data and training arguments for the run
  datasets <- prepare_finetuning_data(
    df = my_data,
    task_type = "classification",
    model_name = MODEL_NAME,
    seed = run_seed
  )

  training_args <- create_training_args(
    output_dir = file.path(OUTPUT_DIR, paste0("run_", i)),
    num_train_epochs = 10,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 64,
    learning_rate = 5e-5,
    task_type = "classification",
    metric_for_best_model = "precision",
    seed = run_seed
  )

  # b. Run the fine-tuning process and store the entire result object
  all_run_results[[paste0("run_", i)]] <- finetune_model(
    datasets = datasets,
    task_type = "classification",
    model_name = MODEL_NAME,
    training_args = training_args,
    num_labels = NUM_LABELS
  )
}



# --- 6. Generate Comprehensive Final Report ---
summarize_run_results(
  all_run_results = all_run_results,
  task_type = "classification",
  label_map = label_map
)


