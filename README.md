# finetuneR: Fine-Tune Transformer Models in R

finetuneR provides a streamlined, high-level workflow for fine-tuning pre-trained transformer models from the Hugging Face library for common text analysis tasks. It handles the Python environment setup, data preparation, model training, and result reporting, allowing R users to leverage state-of-the-art NLP models without leaving their R environment.

## Features

**Automated Setup:** A single function (setup_finetuner_env()) installs all necessary Python packages (transformers, torch, scikit-learn, etc.) into a dedicated reticulate environment.

**Classification & Regression:** Easily fine-tune models for both multi-class text classification and text regression tasks.

**Reproducible Experiments:** Control randomness with seeds at both the global level and for data splitting/training to ensure your results are reproducible.

**User-Friendly Wrappers:** Core functions abstract away the complexities of the underlying Python libraries.

**Comprehensive Reporting:** Generate detailed reports from multiple training runs, including training history, per-class metrics, and an aggregated summary with mean and standard deviation. \## Installation

You can install the development version of finetuneR from [GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("wqu-nd/finetuneR")
# or from devtools
# install.packages("devtools")
devtools::install_github("wqu-nd/finetuneR")
```

## Example

Here is a complete workflow for fine-tuning a text classification model over multiple runs. You can also use the run_finetuning.R file for the classification example.

### 1. Load Libraries

First, load the `finetuneR` package and any other libraries you need.

``` r
library(finetuneR)
library(dplyr)
library(reticulate)
```

### 2. Set Up the Python Environment

This step only needs to be run once per machine or project. It prepares the dedicated Python environment that the package will use.

``` r
# Tell reticulate which Python environment to use
use_miniconda("r-reticulate", required = TRUE)

# Run the setup function to install dependencies and set seeds
setup_finetuner_env(global_seed = 123)
```

### 3. Prepare Your Data

Load your data into a data frame. It must contain a column for your text data and a column for your labels (numeric for both classification and regression).

For this example, we will create a dummy dataset for a 15-class classification problem.

``` r
# Define the mapping from category names to numeric labels
label_map <- data.frame(
  category = c(
    "Monitoring environmental impact", "Preventing pollution", "Strengthening ecosystems",
    "Reducing use", "Reusing", "Recycling", "Repurposing", 
    "Encouraging and supporting others", "Educating and training for sustainability",
    "Creating sustainable products and processes", "Embracing innovation for sustainability",
    "Changing how work is done", "Choosing responsible alternatives",
    "Instituting programs and policies", "Others"
  ),
  label = 0:14
)

# Create some dummy data
my_data <- data.frame(
  text = replicate(1000, paste(sample(letters, 50, replace = TRUE), collapse = "")),
  category = sample(label_map$category, 1000, replace = TRUE)
) %>%
  left_join(label_map, by = "category")



```

### 4. Run the Experiment

Now, you can run your fine-tuning experiment. It's good practice to run the model multiple times with different random seeds to get a more robust estimate of its performance.

``` r
MODEL_NAME <- "distilbert-base-uncased"
NUM_LABELS <- n_distinct(my_data$label)
OUTPUT_DIR <- "./finetuneR-results"

all_run_results <- list()
n_runs <- 3

for (i in 1:n_runs) {
  run_seed <- 42 * i
  cat(paste0("\nExecuting Run ", i, "/", n_runs, " (Seed: ", run_seed, ")..."))
  
  # Prepare datasets with a specific seed for this run's splits
  datasets <- prepare_finetuning_data(
    df = my_data,
    task_type = "classification",
    model_name = MODEL_NAME,
    seed = run_seed
  )
  
  # Create training arguments
  training_args <- create_training_args(
    output_dir = file.path(OUTPUT_DIR, paste0("run_", i)),
    num_train_epochs = 3,
    per_device_train_batch_size = 32,
    seed = run_seed
  )
  
  # Run the fine-tuning process and store the results
  all_run_results[[paste0("run_", i)]] <- finetune_model(
    datasets = datasets,
    task_type = "classification",
    model_name = MODEL_NAME,
    training_args = training_args,
    num_labels = NUM_LABELS
  )
}



```

### 5. Summarize Results

Finally, use the built-in reporting function to view a comprehensive summary of your experiment.

``` r
summarize_run_results(
  all_run_results = all_run_results,
  task_type = "classification",
  label_map = label_map
)



```

This will print the detailed training history and test set evaluation for each run, followed by a final aggregated table showing the mean and standard deviation of the key metrics across all runs.

## License

This package is licensed under the MIT License.
