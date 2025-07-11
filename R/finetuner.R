#' @import reticulate
#' @import dplyr
#' @import R6
#' @importFrom caret createDataPartition
#' @importFrom purrr map_df map_dbl
#' @importFrom tidyr fill

# ==============================================================================
# 1. ENVIRONMENT AND PYTHON SETUP
# ==============================================================================

# Store python modules in a package-local environment to avoid repeated imports
.globals <- new.env(parent = emptyenv())

#' Helper function to check if the Python environment is initialized.
check_env_initialized <- function() {
  if (is.null(.globals$transformers)) {
    stop(
      "The Python environment has not been initialized.\n",
      "Please run `setup_finetuner_env()` before using other package functions.",
      call. = FALSE
    )
  }
}

#' Setup the Python Environment for finetuneR
#'
#' Installs necessary Python packages, imports key libraries, and configures
#' environment variables for parallelism and reproducibility.
#'
#' @param packages A character vector of Python packages to install.
#' @param num_threads An integer specifying the number of threads for libraries
#'   like PyTorch and MKL. Defaults to `1`.
#' @param parallel_tokenizers A logical value. If `FALSE` (the default), it
#'   disables parallelism for Hugging Face tokenizers, which can prevent
#'   deadlocks. Set to `TRUE` to enable it if your environment supports it.
#' @param global_seed An integer to set the master seed for `numpy` and `torch`
#'   for global reproducibility. Defaults to `11`.
#' @export
setup_finetuner_env <- function(packages = c("transformers", "datasets", "torch", "pandas", "numpy", "accelerate", "scikit-learn"),
                                num_threads = 1L,
                                parallel_tokenizers = FALSE,
                                global_seed = 11L) {
  # Set environment variables for torch and transformers.
  # This MUST be done before the library is imported.
  Sys.setenv(
    OMP_NUM_THREADS = as.character(num_threads),
    MKL_NUM_THREADS = as.character(num_threads),
    TOKENIZERS_PARALLELISM = ifelse(parallel_tokenizers, "true", "false")
  )

  # Install packages
  py_install(packages, pip = TRUE)

  # Import core modules into the package environment
  .globals$np <- import("numpy")
  .globals$torch <- import("torch")
  .globals$transformers <- import("transformers")
  .globals$sklearn <- import("sklearn.metrics")

  # Set seeds for reproducibility
  .globals$np$random$seed(as.integer(global_seed))
  .globals$torch$manual_seed(as.integer(global_seed))

  message("Python environment setup complete and modules are loaded.")
}


#' Get Python Dataset Class
#'
#' This internal function defines a Python torch.utils.data.Dataset class
#' using a string.
#' @return A Python class object (`RDataset`).
get_python_dataset_class <- function() {
  if (is.null(.globals$RDataset)) {
    py_run_string("
import torch

class RDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        if isinstance(self.labels[idx], float):
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    ")
    .globals$RDataset <- py$RDataset
  }
  .globals$RDataset
}


# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================

#' Prepare Datasets for Fine-Tuning
#'
#' This function takes a dataframe, splits it into training, validation,
#' and testing sets, tokenizes the text, and wraps the
#' results in Python-compatible Dataset objects ready for the Trainer.
#'
#' @param df A dataframe containing the text and labels.
#' @param task_type A string, either `"classification"` or `"regression"`.
#' @param text_col The name of the column with the text data.
#' @param label_col The name of the column with the label data.
#' @param model_name The name of the pre-trained model from Hugging Face Hub.
#' @param test_split_ratio The proportion of the data to use for the test set.
#' @param val_split_ratio The proportion of the *training* data to use for the
#'   validation set.
#' @param max_length The maximum sequence length for the tokenizer.
#' @param seed An integer for the random seed for reproducibility of the data split.
#' @param stratified A logical value. If `TRUE`, performs stratified sampling
#'   for classification tasks to maintain label distribution. Defaults to `FALSE`.
#' @return A list containing `datasets` (for the trainer) and `data_splits`
#'   (the raw data frames).
#' @export
prepare_finetuning_data <- function(df,
                                    task_type = "classification",
                                    text_col = "text",
                                    label_col = "label",
                                    model_name,
                                    test_split_ratio = 0.2,
                                    val_split_ratio = 0.2,
                                    max_length = 512L,
                                    seed = 11,
                                    stratified = FALSE) {

  # Add check to ensure environment is initialized
  check_env_initialized()

  if (!task_type %in% c("classification", "regression")) {
    stop("`task_type` must be either 'classification' or 'regression'.")
  }

  tokenizer <- .globals$transformers$AutoTokenizer$from_pretrained(model_name)

  # --- Data Splitting ---
  set.seed(seed)
  if (stratified && task_type == "classification") {
    # Stratified split using caret
    train_ids <- createDataPartition(df[[label_col]], p = 1 - test_split_ratio, list = FALSE)
    train_val_df <- df[train_ids, ]
    test_df <- df[-train_ids, ]

    train_ids_2 <- createDataPartition(train_val_df[[label_col]], p = 1 - val_split_ratio, list = FALSE)
    train_df <- train_val_df[train_ids_2, ]
    val_df <- train_val_df[-train_ids_2, ]

  } else {
    # Simple random split
    train_ids <- sample(1:nrow(df), size = floor((1 - test_split_ratio) * nrow(df)))
    train_val_df <- df[train_ids, ]
    test_df <- df[-train_ids, ]

    train_ids_2 <- sample(1:nrow(train_val_df), size = floor((1 - val_split_ratio) * nrow(train_val_df)))
    train_df <- train_val_df[train_ids_2, ]
    val_df <- train_val_df[-train_ids_2, ]
  }

  # --- Tokenize Text and Prepare Labels ---
  X_train <- as.list(train_df[[text_col]])
  X_val <- as.list(val_df[[text_col]])
  X_test <- as.list(test_df[[text_col]])

  if (task_type == "classification") {
    Y_train <- as.list(as.integer(train_df[[label_col]]))
    Y_val <- as.list(as.integer(val_df[[label_col]]))
    Y_test <- as.list(as.integer(test_df[[label_col]]))
  } else { # Regression
    Y_train <- as.list(as.numeric(train_df[[label_col]]))
    Y_val <- as.list(as.numeric(val_df[[label_col]]))
    Y_test <- as.list(as.numeric(test_df[[label_col]]))
  }

  train_enc <- tokenizer(X_train, truncation = TRUE, padding = TRUE, max_length = max_length)
  val_enc <- tokenizer(X_val, truncation = TRUE, padding = TRUE, max_length = max_length)
  test_enc <- tokenizer(X_test, truncation = TRUE, padding = TRUE, max_length = max_length)

  RDataset <- get_python_dataset_class()

  train_py_dataset <- RDataset(train_enc, r_to_py(Y_train))
  val_py_dataset <- RDataset(val_enc, r_to_py(Y_val))
  test_py_dataset <- RDataset(test_enc, r_to_py(Y_test))

  # Return both the tokenized datasets and the raw data frames
  list(
    datasets = list(
      train = train_py_dataset,
      validation = val_py_dataset,
      test = test_py_dataset
    ),
    data_splits = list(
      train = train_df,
      validation = val_df,
      test = test_df
    )
  )
}

# ==============================================================================
# 3. TRAINING AND EVALUATION
# ==============================================================================

#' Create User-Friendly Training Arguments
#'
#' A helper function to simplify the creation of `TrainingArguments`.
#'
#' @param output_dir Path to the directory where model checkpoints will be saved.
#' @param num_train_epochs Number of times to iterate over the training dataset.
#' @param learning_rate The initial learning rate for the AdamW optimizer.
#' @param per_device_train_batch_size The batch size for training.
#' @param per_device_eval_batch_size The batch size for validation.
#' @param warmup_steps Number of steps for the learning rate warmup. Defaults to 0.
#' @param weight_decay The weight decay to apply (if not zero). Defaults to 0.
#' @param metric_for_best_model The metric used to identify the best model.
#'   Defaults to "f1" for classification and "mse" for regression.
#' @param eval_strategy The evaluation and save strategy to adopt during training.
#'   Possible values are `"no"`, `"steps"`, `"epoch"`. Defaults to `"epoch"`.
#' @param logging_strategy The logging strategy to adopt during training.
#'   Possible values are `"no"`, `"steps"`, `"epoch"`. Defaults to `"epoch"`.
#' @param show_progress_bar A logical value. If `FALSE`, disables the detailed,
#'   per-step progress bar. Defaults to `TRUE`.
#' @param task_type A string to set a default for `metric_for_best_model`.
#' @param seed A random seed for reproducibility for the training process.
#' @param ... Other arguments to be passed directly to `transformers$TrainingArguments`.
#' @return A `TrainingArguments` object.
#' @export
create_training_args <- function(output_dir,
                                 num_train_epochs = 3L,
                                 learning_rate = 2e-5,
                                 per_device_train_batch_size = 16L,
                                 per_device_eval_batch_size = 32L,
                                 warmup_steps = 0L,
                                 weight_decay = 0.01,
                                 metric_for_best_model = NULL,
                                 eval_strategy = "epoch",
                                 logging_strategy = "epoch",
                                 show_progress_bar = TRUE,
                                 task_type = "classification",
                                 seed = 11,
                                 ...) {

  check_env_initialized()

  if (is.null(metric_for_best_model)) {
    metric_for_best_model <- ifelse(task_type == "classification", "f1", "mse")
  }

  # Pass arguments directly, mapping user-friendly names to Python library's names
  .globals$transformers$TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = as.integer(num_train_epochs),
    learning_rate = learning_rate,
    per_device_train_batch_size = as.integer(per_device_train_batch_size),
    per_device_eval_batch_size = as.integer(per_device_eval_batch_size),
    warmup_steps = as.integer(warmup_steps),
    weight_decay = weight_decay,
    eval_strategy = eval_strategy,
    save_strategy = eval_strategy, # Tie save strategy to evaluation strategy
    load_best_model_at_end = TRUE,
    metric_for_best_model = metric_for_best_model,
    logging_strategy = logging_strategy,
    disable_tqdm = !show_progress_bar,
    seed = as.integer(seed),
    ...
  )
}


#' Fine-Tune a Sequence Model
#'
#' Orchestrates the fine-tuning process for classification or regression.
#'
#' @param datasets A list of datasets created by `prepare_finetuning_data`.
#' @param task_type A string, either `"classification"` or `"regression"`.
#' @param model_name The name of the pre-trained model from Hugging Face Hub.
#' @param training_args A `TrainingArguments` object from `transformers`.
#' @param num_labels For classification, the number of unique classes. For
#'   regression, this is automatically set to 1 and can be omitted.
#' @param use_cuda If `TRUE`, will attempt to use a GPU if available.
#' @return A list with the trained `Trainer` and prediction output.
#' @export
finetune_model <- function(datasets,
                           task_type = "classification",
                           model_name,
                           training_args,
                           num_labels = NULL,
                           use_cuda = FALSE) {

  check_env_initialized()

  if (!task_type %in% c("classification", "regression")) {
    stop("`task_type` must be either 'classification' or 'regression'.")
  }

  if (task_type == "classification") {
    if (is.null(num_labels)) stop("`num_labels` must be provided for classification tasks.")

    compute_metrics <- function(eval_pred) {
      logits <- eval_pred$predictions; labels <- eval_pred$label_ids
      preds <- apply(py_to_r(logits), 1, which.max) - 1L
      labs <- as.integer(py_to_r(labels))

      list(
        precision = .globals$sklearn$precision_score(labs, preds, average = "weighted", zero_division = 0),
        recall = .globals$sklearn$recall_score(labs, preds, average = "weighted", zero_division = 0),
        f1 = .globals$sklearn$f1_score(labs, preds, average = "weighted", zero_division = 0)
      )
    }
  } else { # Regression
    num_labels <- 1L

    compute_metrics <- function(eval_pred) {
      logits <- eval_pred$predictions; labels <- eval_pred$label_ids
      preds <- py_to_r(logits)[, 1]
      labs <- as.numeric(py_to_r(labels))

      # Check for zero variance in predictions before calculating correlation
      if (stats::sd(preds) == 0) {
        pearson_r <- NA_real_
      } else {
        pearson_r <- .globals$np$corrcoef(labs, preds)[0, 1]
      }

      list(
        mae = .globals$sklearn$mean_absolute_error(labs, preds),
        mse = .globals$sklearn$mean_squared_error(labs, preds),
        pearson_r = pearson_r
      )
    }
  }

  model <- .globals$transformers$AutoModelForSequenceClassification$from_pretrained(
    model_name, num_labels = num_labels
  )

  training_args$no_cuda <- !use_cuda

  trainer <- .globals$transformers$Trainer(
    model = model,
    args = training_args,
    train_dataset = datasets$train,
    eval_dataset = datasets$validation,
    compute_metrics = compute_metrics
  )

  trainer$train()

  list(
    trainer = trainer,
    prediction_output = trainer$predict(datasets$test)
  )
}

#' Evaluate Model Predictions
#'
#' Generates an evaluation report from a model's prediction output.
#'
#' @param prediction_output The output object from a `trainer$predict()` call.
#' @param task_type A string, either `"classification"` or `"regression"`.
#' @return For classification, a classification report. For regression, a
#'   list of metrics (MAE, MSE, RMSE, Pearson's R).
#' @export
evaluate_predictions <- function(prediction_output, task_type = "classification") {

  check_env_initialized()

  if (!task_type %in% c("classification", "regression")) {
    stop("`task_type` must be either 'classification' or 'regression'.")
  }

  y_true <- prediction_output$label_ids

  if (task_type == "classification") {
    y_pred <- .globals$np$argmax(prediction_output$predictions, axis = -1L)
    .globals$sklearn$classification_report(y_true, y_pred, output_dict = TRUE, zero_division = 0)
  } else { # Regression
    y_pred <- py_to_r(prediction_output$predictions)[, 1]

    pearson_r <- tryCatch({
      if (stats::sd(y_pred) < 1e-9 || stats::sd(y_true) < 1e-9) {
        return(NA_real_)
      }
      val <- .globals$np$corrcoef(y_true, y_pred)[0, 1]
      if (is.numeric(val) && length(val) == 1 && !is.nan(val)) val else NA_real_
    }, error = function(e) {
      NA_real_
    })

    list(
      mae = .globals$sklearn$mean_absolute_error(y_true, y_pred),
      mse = .globals$sklearn$mean_squared_error(y_true, y_pred),
      rmse = sqrt(.globals$sklearn$mean_squared_error(y_true, y_pred)),
      pearson_r = pearson_r
    )
  }
}


#' Format Trainer Log History
#'
#' Converts the `log_history` from a trained `Trainer` object into a tidy
#' data frame for easier analysis and plotting. It now correctly merges
#' training loss with validation metrics for a clean, epoch-by-epoch summary.
#'
#' @param log_history The `$state$log_history` object from a trained trainer.
#' @return A data frame with columns for epoch, training loss, validation loss,
#'   and other core validation metrics.
#' @export
format_log_history <- function(log_history) {
  # Convert the Python list of dictionaries to an R list of lists
  logs_r <- py_to_r(log_history)

  # Process logs into a more structured list
  processed_logs <- list()
  current_train_loss <- NA_real_

  for (log in logs_r) {
    if ("loss" %in% names(log)) {
      # This is a training log, update the last known training loss
      current_train_loss <- log$loss
    }
    if ("eval_loss" %in% names(log)) {
      # This is a validation log, add the last training loss to it
      log$training_loss <- current_train_loss

      # Sanitize the log by replacing any NULLs with NA before appending
      sanitized_log <- lapply(log, function(x) {
        if (is.null(x) || length(x) == 0) {
          NA_real_
        } else {
          x
        }
      })

      processed_logs <- append(processed_logs, list(sanitized_log))
    }
  }

  if (length(processed_logs) == 0) return(data.frame())

  # Now this call is safe because all lists have elements of length 1
  map_df(processed_logs, ~ as.data.frame(.x)) %>%
    rename_with(~gsub("eval_", "validation_", .x)) %>%
    select(
      any_of(c("epoch", "step", "training_loss", "validation_loss",
               "validation_precision", "validation_recall", "validation_f1",
               "validation_mae", "validation_mse", "validation_pearson_r"))
    )
}

# ==============================================================================
# 4. REPORTING
# ==============================================================================

#' Summarize Results from Multiple Runs
#'
#' Prints a comprehensive report detailing the results for each run of a
#' multi-run experiment and concludes with an aggregated summary table.
#'
#' @param all_run_results A list where each element is the output from a
#'   `finetune_model()` call. The list should be named by run (e.g., "run_1").
#' @param task_type A string, either `"classification"` or `"regression"`.
#' @param label_map A dataframe for classification tasks that maps the numeric
#'   `label` column to a human-readable `category` column.
#' @return Invisibly returns a data frame containing the aggregated summary
#'   statistics.
#' @export
summarize_run_results <- function(all_run_results, task_type = "classification", label_map = NULL) {

    check_env_initialized()
    final_reports <- list()

    cat("\n\n===== COMPREHENSIVE RESULTS REPORT =====\n")

    for (i in seq_along(all_run_results)) {
      run_name <- names(all_run_results)[i]
      run_result <- all_run_results[[i]]

      cat(paste0("\n--- Results for ", run_name, " ---\n"))

      # a. Training History
      history <- format_log_history(run_result$trainer$state$log_history)
      cat("Training History:\n")
      print(history, width = 120)

      # b. Runtime
      full_history <- py_to_r(run_result$trainer$state$log_history)
      runtime_secs <- full_history[[length(full_history)]]$train_runtime
      cat(paste("\nRuntime:", round(runtime_secs / 60, 2), "minutes\n"))

      # c. Test Set Evaluation
      test_report <- evaluate_predictions(
        prediction_output = run_result$prediction_output,
        task_type = task_type
      )
      cat("Test Set Evaluation:\n")
      if (task_type == "classification") {
        # Separate per-class metrics from the summary metrics to avoid format issues
        class_keys <- names(test_report)[!names(test_report) %in% c("accuracy", "macro avg", "weighted avg")]

        # Robustly build the data frame, handling NULLs
        report_rows <- list()
        for(class_name in class_keys) {
          metrics <- test_report[[class_name]]
          # Sanitize each metric in the list, replacing NULL with NA
          sanitized_metrics <- lapply(metrics, function(val) if(is.null(val)) NA else val)
          report_rows[[class_name]] <- as.data.frame(sanitized_metrics)
        }
        report_df <- bind_rows(report_rows, .id = "class") %>%
          rename(f1_score = `f1.score`) # fix name for R

        if (!is.null(label_map)) {
          # Add human-readable names if map is provided
          report_df <- report_df %>%
            mutate(class = as.integer(class)) %>%
            left_join(label_map, by = c("class" = "label")) %>%
            select(category, class, precision, recall, f1_score, support)
        }

        print(report_df)

        cat("\nSummary Metrics:\n")
        print(data.frame(test_report$`weighted avg`))

        final_reports[[run_name]] <- list(
          test_metrics = test_report,
          runtime_secs = runtime_secs
        )
      } else {
        print(as.data.frame(test_report))
        final_reports[[run_name]] <- list(
          test_metrics = test_report,
          runtime_secs = runtime_secs
        )
      }
    }

    # --- Aggregate and Print Final Summary Table ---
    cat("\n\n===== FINAL AGGREGATED SUMMARY =====\n")

    if (task_type == "classification") {
      summary_data <- purrr::map_df(final_reports, ~ {
        # Use [[...]] to access names with hyphens
        data.frame(
          precision = .x$test_metrics$`weighted avg`$precision,
          recall = .x$test_metrics$`weighted avg`$recall,
          f1.score = .x$test_metrics$`weighted avg`$`f1-score`
        )
      })

      final_summary <- summary_data %>%
        summarise(
          mean_precision = mean(precision), sd_precision = sd(precision),
          mean_recall = mean(recall), sd_recall = sd(recall),
          mean_f1 = mean(f1.score), sd_f1 = sd(f1.score)
        )
    } else { # Regression
      summary_data <- purrr::map_df(final_reports, ~ as.data.frame(.x$test_metrics))
      final_summary <- summary_data %>%
        summarise(
          mean_mae = mean(mae), sd_mae = sd(mae),
          mean_mse = mean(mse), sd_mse = sd(mse),
          mean_rmse = mean(rmse), sd_rmse = sd(rmse),
          mean_pearson_r = mean(pearson_r), sd_pearson_r = sd(pearson_r)
        )
    }

    # Add timing summary
    runtimes <- purrr::map_dbl(final_reports, "runtime_secs")
    final_summary$mean_runtime_mins <- mean(runtimes) / 60
    final_summary$total_runtime_mins <- sum(runtimes) / 60

    print(final_summary)

    invisible(final_summary)
}
