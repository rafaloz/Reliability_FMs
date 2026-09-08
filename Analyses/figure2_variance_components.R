#!/usr/bin/env Rscript

# Trace-pooled variance decomposition for Figure 2.
#
# Acquisition 001 is retained in every participant-scanner cell. Every
# representation coordinate is fitted with the crossed random-intercept model
#
#   value ~ 1 + (1 | subject_id) + (1 | scanner)
#
# using REML. Responses are standardized only for numerical fitting and the
# estimated variances are returned to their native scale before summation.
# Consequently, the reported subject, scanner, and residual components are
# traces (sums of marginal variances), not means of coordinate-wise percentages.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
  library(parallel)
})

parse_args <- function(argv) {
  values <- list(
    input = NULL,
    input_sha256 = NULL,
    output = NULL,
    condition = NULL,
    workers = 4L
  )
  keys <- c(
    "--input" = "input",
    "--input-sha256" = "input_sha256",
    "--output" = "output",
    "--condition" = "condition",
    "--workers" = "workers"
  )
  index <- 1L
  while (index <= length(argv)) {
    key <- argv[[index]]
    if (!key %in% names(keys) || index == length(argv)) {
      stop("Unknown option or missing value: ", key)
    }
    values[[keys[[key]]]] <- argv[[index + 1L]]
    index <- index + 2L
  }
  values$workers <- as.integer(values$workers)
  if (is.null(values$input) || is.null(values$input_sha256) ||
      is.null(values$output) ||
      is.null(values$condition)) {
    stop(paste0(
      "Required: --input CSV --input-sha256 SHA256 --output JSON ",
      "--condition NAME"
    ))
  }
  if (!grepl("^[0-9a-f]{64}$", values$input_sha256)) {
    stop("--input-sha256 must contain exactly 64 lowercase hexadecimal digits")
  }
  if (!is.finite(values$workers) || values$workers < 1L) {
    stop("--workers must be a positive integer")
  }
  values
}

numeric_suffix <- function(name) {
  as.integer(sub(".*_", "", name))
}

select_features <- function(frame) {
  features <- grep("^emb_[0-9]+$", names(frame), value = TRUE)
  features <- features[order(numeric_suffix(features))]
  if (length(features) == 0L) stop("No emb_<index> columns found")
  expected <- seq.int(0L, length(features) - 1L)
  observed <- unname(vapply(features, numeric_suffix, integer(1)))
  if (!identical(observed, expected)) {
    stop("Embedding columns are not contiguous from emb_0")
  }
  features
}

select_primary_cells <- function(frame, features) {
  required <- c("subject_id", "session_id", "scanner", features)
  missing <- setdiff(required, names(frame))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
  if (anyNA(frame[, c("subject_id", "scanner")]) ||
      any(frame$subject_id == "") || any(frame$scanner == "")) {
    stop("Missing participant or scanner labels")
  }
  values <- as.matrix(frame[, features, drop = FALSE])
  storage.mode(values) <- "double"
  if (any(!is.finite(values))) stop("Non-finite representation values")
  n_cells <- nrow(unique(frame[, c("subject_id", "scanner")]))
  selected <- frame[grepl("001$", as.character(frame$session_id)), , drop = FALSE]
  if (nrow(selected) != n_cells ||
      any(duplicated(selected[, c("subject_id", "scanner")]))) {
    stop("Every participant-scanner cell must contain exactly one acquisition 001")
  }
  selected[, c("subject_id", "scanner", features), drop = FALSE]
}

lmer_control <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 100000L),
  calc.derivs = TRUE,
  check.conv.singular = "ignore"
)

fit_coordinate <- function(feature, cells) {
  response <- as.numeric(cells[[feature]])
  response_sd <- sd(response)
  if (!is.finite(response_sd) || response_sd <= .Machine$double.eps) {
    return(c(
      subject = NA_real_, scanner = NA_real_, residual = NA_real_,
      singular = NA_real_, converged = 0, scale = response_sd
    ))
  }
  fit_data <- data.frame(
    response = (response - mean(response)) / response_sd,
    subject_id = factor(cells$subject_id),
    scanner = factor(cells$scanner)
  )
  fit <- tryCatch(
    suppressMessages(suppressWarnings(
      lmer(
        response ~ 1 + (1 | subject_id) + (1 | scanner),
        data = fit_data,
        REML = TRUE,
        control = lmer_control
      )
    )),
    error = function(error) NULL
  )
  if (is.null(fit)) {
    return(c(
      subject = NA_real_, scanner = NA_real_, residual = NA_real_,
      singular = NA_real_, converged = 0, scale = response_sd
    ))
  }
  optimizer_code <- fit@optinfo$conv$opt
  messages <- fit@optinfo$conv$lme4$messages
  converged <- (is.null(optimizer_code) || as.integer(optimizer_code) == 0L) &&
    (is.null(messages) || length(messages) == 0L)
  if (!converged) {
    return(c(
      subject = NA_real_, scanner = NA_real_, residual = NA_real_,
      singular = as.numeric(isSingular(fit, tol = 1e-5)),
      converged = 0, scale = response_sd
    ))
  }
  components <- as.data.frame(VarCorr(fit))
  get_component <- function(group) {
    value <- components$vcov[components$grp == group]
    if (length(value) != 1L) NA_real_ else as.numeric(value)
  }
  restore <- response_sd ^ 2
  c(
    subject = get_component("subject_id") * restore,
    scanner = get_component("scanner") * restore,
    residual = get_component("Residual") * restore,
    singular = as.numeric(isSingular(fit, tol = 1e-5)),
    converged = 1,
    scale = response_sd
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (!file.exists(args$input)) stop("Input not found: ", args$input)
dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)

raw <- read.csv(args$input, check.names = FALSE)
features <- select_features(raw)
cells <- select_primary_cells(raw, features)

workers <- min(args$workers, length(features))
fitted <- mclapply(
  features,
  fit_coordinate,
  cells = cells,
  mc.cores = workers,
  mc.preschedule = TRUE
)
components <- do.call(rbind, fitted)
rownames(components) <- features

valid <- components[, "converged"] == 1 &
  apply(components[, c("subject", "scanner", "residual"), drop = FALSE],
        1L, function(values) all(is.finite(values) & values >= 0))
if (!any(valid)) stop("No coordinate produced valid REML components")

trace_subject <- sum(components[valid, "subject"])
trace_scanner <- sum(components[valid, "scanner"])
trace_residual <- sum(components[valid, "residual"])
trace_total <- trace_subject + trace_scanner + trace_residual
if (!is.finite(trace_total) || trace_total <= 0) stop("Invalid pooled trace")

payload <- list(
  schema_version = 1L,
  condition = args$condition,
  input_csv = normalizePath(args$input),
  input_csv_sha256 = args$input_sha256,
  estimator = "REML via lme4::lmer",
  formula = "value ~ 1 + (1 | subject_id) + (1 | scanner)",
  repeat_handling = paste0(
    "acquisition 001 retained in each participant-scanner cell before fitting"
  ),
  response_scaling = paste0(
    "per-coordinate z-score for numerical fitting; variance components ",
    "multiplied by the original coordinate variance before trace pooling"
  ),
  pooling = paste0(
    "sum native-scale marginal variance components across valid coordinates; ",
    "proportions are component trace divided by total trace"
  ),
  n_sessions_input = nrow(raw),
  n_subject_scanner_cells = nrow(cells),
  n_subjects = length(unique(cells$subject_id)),
  n_scanners = length(unique(cells$scanner)),
  n_dimensions = length(features),
  n_dimensions_valid = sum(valid),
  n_dimensions_failed_or_constant = sum(!valid),
  n_singular = sum(components[, "singular"] == 1, na.rm = TRUE),
  trace_subject = trace_subject,
  trace_scanner = trace_scanner,
  trace_residual = trace_residual,
  trace_total = trace_total,
  proportion_subject = trace_subject / trace_total,
  proportion_scanner = trace_scanner / trace_total,
  proportion_residual = trace_residual / trace_total,
  software = list(
    R = R.version.string,
    lme4 = as.character(packageVersion("lme4")),
    jsonlite = as.character(packageVersion("jsonlite"))
  )
)

temporary <- paste0(args$output, ".tmp")
write_json(
  payload,
  temporary,
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = NA,
  na = "null"
)
if (!file.rename(temporary, args$output)) {
  unlink(temporary)
  stop("Could not atomically publish output: ", args$output)
}
