% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/finetuner.R
\name{evaluate_predictions}
\alias{evaluate_predictions}
\title{Evaluate Model Predictions}
\usage{
evaluate_predictions(prediction_output, task_type = "classification")
}
\arguments{
\item{prediction_output}{The output object from a `trainer$predict()` call.}

\item{task_type}{A string, either `"classification"` or `"regression"`.}
}
\value{
For classification, a classification report. For regression, a
  list of metrics (MAE, MSE, RMSE, Pearson's R).
}
\description{
Generates an evaluation report from a model's prediction output.
}
