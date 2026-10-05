# ---------------------------------------------------------------------------
# Simulation from a fitted hhh4, for the component-recovery study. Called from
# Python (src/experiments/recovery.py):
#
#     Rscript hhh4_simulate.R <folder>
#
# Inputs as for hhh4_fit.R (counts.csv, adjacency.csv, population.csv,
# spec.csv) plus in spec.csv:
#   scenarios   space-separated names, each one of
#                 fitted     - parameters as estimated
#                 no_ne      - no neighbourhood transmission (own rate as fitted,
#                              so epidemics are smaller)
#                 strong_ne  - 75% of the transmission via neighbours, 25% own,
#                              same total rate as fitted
#               A scenario whose simulation explodes is skipped with a message.
#                 no_season  - no seasonality in the epidemic / neighbourhood rates
#   sim_seed    random seed
#
# Outputs per scenario <s>:
#   sim_<s>_counts.csv       date, node_0..node_{N-1}  simulated weekly counts
#   sim_<s>_components.csv   row, node, endemic, epidemic, neighbourhood, mean:
#                            the TRUE one-step components of the simulated series
#   sim_coefficients.csv     the fitted coefficients used as truth (per scenario)
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(surveillance))

args   <- commandArgs(trailingOnly = TRUE)
folder <- if (length(args)) args[1] else "."
inp    <- function(f) file.path(folder, f)
s      <- read.csv(inp("spec.csv"), stringsAsFactors = FALSE); spec <- setNames(as.list(s$value), s$key)
num    <- function(k, d) if (!is.null(spec[[k]]) && nzchar(spec[[k]])) as.numeric(spec[[k]]) else d
flag   <- function(k, d) if (!is.null(spec[[k]])) toupper(spec[[k]]) %in% c("TRUE", "1", "YES") else d

fit_end   <- num("fit_end_row", NA)
S         <- num("harmonics", 1)
max_lag   <- num("max_lag", 5)
use_ri    <- flag("random_effects", TRUE)
family    <- if (!is.null(spec$family)) spec$family else "NegBin1"
scenarios <- strsplit(if (!is.null(spec$scenarios)) spec$scenarios else "fitted no_ne", " ")[[1]]
set.seed(num("sim_seed", 1))

counts <- read.csv(inp("counts.csv"), check.names = FALSE)
dates  <- as.Date(counts[[1]])
Y      <- as.matrix(counts[, -1, drop = FALSE]); storage.mode(Y) <- "integer"
N      <- ncol(Y); nT <- nrow(Y)
A      <- as.matrix(read.csv(inp("adjacency.csv"), header = FALSE))
A      <- (A + t(A)) > 0; diag(A) <- FALSE; storage.mode(A) <- "integer"
pop    <- read.csv(inp("population.csv"))
popfrac <- matrix(pop$population / sum(pop$population), nrow = nT, ncol = N, byrow = TRUE)
nb <- nbOrder(A, maxlag = max_lag); dimnames(nb) <- list(colnames(Y), colnames(Y))
sts_obj <- sts(observed = Y, start = c(as.integer(format(dates[1], "%Y")), 1), frequency = 52,
               neighbourhood = nb, population = popfrac)

f <- function() {
  base <- if (use_ri) ~ -1 + ri(type = "iid", corr = "all") else ~ 1
  if (S > 0) addSeason2formula(base, S = S, period = 52) else base
}
ctrl <- list(end = list(f = f(), offset = population(sts_obj)), ar = list(f = f()),
             ne = list(f = f(), weights = W_powerlaw(maxlag = max_lag, normalize = TRUE, log = TRUE)),
             family = family, subset = 2:fit_end)
fit <- tryCatch(hhh4(sts_obj, ctrl), error = function(e) e)
if (inherits(fit, "error") || !isTRUE(fit$convergence)) {
  use_ri <- FALSE
  ctrl$end$f <- update(f(), ~ . + fe(1, unitSpecific = TRUE) - 1)
  ctrl$ar$f <- f(); ctrl$ne$f <- f()
  fit <- hhh4(sts_obj, ctrl)
}

theta0 <- fit$coefficients
coefs_out <- list()
for (sc in scenarios) {
  theta <- theta0
  nm <- names(theta)
  int_ar <- grep("^ar\\.(ri\\(iid\\)|\\(Intercept\\)|1)$", nm)
  int_ne <- grep("^ne\\.(ri\\(iid\\)|\\(Intercept\\)|1)$", nm)
  total <- exp(theta[int_ar]) + exp(theta[int_ne])
  if (sc == "no_ne")     theta[int_ne] <- -30      # own rate as fitted (adding the
                                                     # neighbour rate to it can explode)
  if (sc == "strong_ne") { theta[int_ar] <- log(0.25 * total); theta[int_ne] <- log(0.75 * total) }
  if (sc == "no_season") theta[grep("^(ar|ne)\\.(sin|cos)", nm)] <- 0
  fit_s <- fit; fit_s$coefficients <- theta
  coefs_out[[sc]] <- data.frame(scenario = sc, name = nm[seq_len(fit$dim[1])], value = theta[seq_len(fit$dim[1])])

  sim <- simulate(fit_s, nsim = 1, seed = NULL, subset = 2:nT, y.start = Y[1, , drop = FALSE], simplify = TRUE)
  ysim <- rbind(Y[1, ], matrix(sim[, , 1], nrow = nT - 1))
  if (any(!is.finite(ysim)) || max(ysim) > 1e8) {
    message("scenario ", sc, " explodes (supercritical); skipped")
    next
  }
  storage.mode(ysim) <- "integer"

  # true one-step components of the simulated series: rebuild the model terms on
  # the simulated data (lagged counts), then evaluate them at the true parameters
  sts_sim <- sts_obj; observed(sts_sim) <- ysim
  fit_sim <- fit_s; fit_sim$stsObj <- sts_sim; fit_sim$terms <- NULL
  mm <- meanHHH(theta, terms(fit_sim), subset = 2:nT)
  comp <- data.frame(row = rep(2:nT, times = N), node = rep(seq_len(N) - 1, each = nT - 1),
                     endemic = as.vector(mm$endemic), epidemic = as.vector(mm$epi.own),
                     neighbourhood = as.vector(mm$epi.neighbours), mean = as.vector(mm$mean))

  out <- data.frame(date = format(dates, "%Y-%m-%d"), ysim); names(out)[-1] <- colnames(Y)
  write.csv(out, inp(paste0("sim_", sc, "_counts.csv")), row.names = FALSE)
  write.csv(comp, inp(paste0("sim_", sc, "_components.csv")), row.names = FALSE)
}
write.csv(do.call(rbind, coefs_out), inp("sim_coefficients.csv"), row.names = FALSE)
cat("simulated scenarios:", paste(scenarios, collapse = ", "), "\n")
