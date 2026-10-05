# ---------------------------------------------------------------------------
# hhh4 reference model (R package 'surveillance'), called from Python by
# src/experiments/hhh4r.py. Do not run by hand unless debugging:
#
#     Rscript hhh4_fit.R <folder>
#
# <folder> holds the inputs written by Python:
#   counts.csv      date, node_0 ... node_{N-1}   (weekly counts, all weeks)
#   adjacency.csv   N x N 0/1 matrix, no header   (who borders whom)
#   population.csv  node, population
#   spec.csv        key, value                    (see read_spec below)
#
# and receives the outputs:
#   predictions.csv  per forecast origin (t0) and node: lead-L quantiles from
#                    simulating the fitted model forward, the empirical CDF at
#                    the truth (for the randomised PIT) and the simulated mean
#   components.csv   one-step-ahead components at each target week, given the
#                    observed past: endemic, epidemic (own), neighbourhood, mean
#   coefficients.csv fixed effects with standard errors
#   unit_effects.csv per node: endemic / epidemic / neighbourhood log-effects
#   fit_info.csv     log-likelihood, convergence, run time, specification used
#
# Model (Meyer, Held & Hoehle 2017, J Stat Softw; Paul & Held 2011):
#   mu_it = e_it nu_it + lambda_it y_i,t-1 + phi_it sum_j w_ji y_j,t-1
#   each log-predictor: unit random intercept + S harmonics of week of year;
#   neighbourhood weights: power law in neighbourhood order; NegBin family.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(surveillance))

args   <- commandArgs(trailingOnly = TRUE)
folder <- if (length(args)) args[1] else "."
inp    <- function(f) file.path(folder, f)

read_spec <- function(path) {
  s <- read.csv(path, stringsAsFactors = FALSE)
  setNames(as.list(s$value), s$key)
}
spec <- read_spec(inp("spec.csv"))
num  <- function(k, d) if (!is.null(spec[[k]]) && nzchar(spec[[k]])) as.numeric(spec[[k]]) else d
vec  <- function(k) as.numeric(strsplit(spec[[k]], " ")[[1]])
flag <- function(k, d) if (!is.null(spec[[k]])) toupper(spec[[k]]) %in% c("TRUE", "1", "YES") else d

fit_end   <- num("fit_end_row", NA)
t0_rows   <- vec("t0_rows")
lead      <- num("lead", 1)
quantiles <- vec("quantiles")
nsim      <- num("nsim", 500)
seed      <- num("seed", 1)
S         <- num("harmonics", 1)
max_lag   <- num("max_lag", 5)
use_ri    <- flag("random_effects", TRUE)
power_law <- flag("power_law", TRUE)
family    <- if (!is.null(spec$family)) spec$family else "NegBin1"

counts <- read.csv(inp("counts.csv"), check.names = FALSE)
dates  <- as.Date(counts[[1]])
Y      <- as.matrix(counts[, -1, drop = FALSE]); storage.mode(Y) <- "integer"
N      <- ncol(Y)
A      <- as.matrix(read.csv(inp("adjacency.csv"), header = FALSE))
A      <- (A + t(A)) > 0; diag(A) <- FALSE; storage.mode(A) <- "integer"
pop    <- read.csv(inp("population.csv"))
popfrac <- matrix(pop$population / sum(pop$population), nrow = nrow(Y), ncol = N, byrow = TRUE)

nb <- if (power_law) nbOrder(A, maxlag = max_lag) else A
dimnames(nb) <- list(colnames(Y), colnames(Y))
sts_obj <- sts(observed = Y, start = c(as.integer(format(dates[1], "%Y")), 1), frequency = 52,
               neighbourhood = nb, population = popfrac)

ri_f <- function(intercept_only = FALSE) {
  base <- if (use_ri) ~ -1 + ri(type = "iid", corr = "all") else ~ 1
  if (S > 0) addSeason2formula(base, S = S, period = 52) else base
}
weights <- if (power_law) W_powerlaw(maxlag = max_lag, normalize = TRUE, log = TRUE) else neighbourhood(sts_obj) == 1

make_control <- function() list(
  end = list(f = ri_f(), offset = population(sts_obj)),
  ar  = list(f = ri_f()),
  ne  = list(f = ri_f(), weights = weights),
  family = family,
  subset = 2:fit_end)

t_start <- Sys.time()
fit <- tryCatch(hhh4(sts_obj, make_control()), error = function(e) e)
if (inherits(fit, "error") || !isTRUE(fit$convergence)) {
  message("hhh4 with random effects failed or did not converge; refitting with unit-specific fixed endemic intercepts")
  use_ri <- FALSE
  ctrl <- make_control()
  ctrl$end$f <- update(ctrl$end$f, ~ . + fe(1, unitSpecific = TRUE) - 1)
  fit <- hhh4(sts_obj, ctrl)
}
runtime <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

# ---- lead-L forecasts by simulation from each origin -------------------------
qcols <- paste0("q_", seq_along(quantiles))
rows  <- vector("list", length(t0_rows))
set.seed(seed)
for (k in seq_along(t0_rows)) {
  t0 <- t0_rows[k]
  tgt <- t0 + lead
  if (tgt > nrow(Y)) next
  sims <- simulate(fit, nsim = nsim, seed = NULL, subset = (t0 + 1):tgt,
                   y.start = Y[t0, , drop = FALSE])
  draws <- matrix(sims[lead, , ], nrow = N)            # N x nsim at the target week
  q  <- t(apply(draws, 1, quantile, probs = quantiles, type = 1))
  y  <- Y[tgt, ]
  hi <- rowMeans(draws <= y)
  lo <- rowMeans(draws <= y - 1)
  df <- data.frame(t0_row = t0, node = seq_len(N) - 1, target = y, q, cdf_lo = lo, cdf_hi = hi,
                   mean_sim = rowMeans(draws))
  names(df)[4:(3 + length(quantiles))] <- qcols
  rows[[k]] <- df
}
write.csv(do.call(rbind, rows), inp("predictions.csv"), row.names = FALSE)

# ---- one-step components at the target weeks ---------------------------------
tgt_rows <- sort(unique(pmin(t0_rows + lead, nrow(Y))))
mm <- meanHHH(fit$coefficients, terms(fit), subset = tgt_rows)
comp <- data.frame(
  row       = rep(tgt_rows, times = N),
  node      = rep(seq_len(N) - 1, each = length(tgt_rows)),
  endemic   = as.vector(mm$endemic),
  epidemic  = as.vector(mm$epi.own),
  neighbourhood = as.vector(mm$epi.neighbours),
  mean      = as.vector(mm$mean),
  target    = as.vector(Y[tgt_rows, , drop = FALSE]))
write.csv(comp, inp("components.csv"), row.names = FALSE)

# ---- parameters ----------------------------------------------------------------
co <- summary(fit)$fixef
write.csv(data.frame(name = rownames(co), estimate = co[, 1], se = co[, 2]),
          inp("coefficients.csv"), row.names = FALSE)

if (use_ri) {
  re <- ranef(fit, tomatrix = TRUE)
  ue <- data.frame(node = seq_len(N) - 1, re, check.names = FALSE)
} else {
  ue <- data.frame(node = seq_len(N) - 1)
}
write.csv(ue, inp("unit_effects.csv"), row.names = FALSE)

write.csv(data.frame(
  key = c("loglik", "converged", "runtime_s", "random_effects", "family", "harmonics",
          "power_law", "fit_end_row", "nsim", "power_law_d"),
  value = c(as.numeric(logLik(fit)), isTRUE(fit$convergence), runtime, use_ri, family, S,
            power_law, fit_end, nsim,
            if (power_law) exp(coef(fit)[grep("neweights", names(coef(fit)))[1]]) else NA)),
  inp("fit_info.csv"), row.names = FALSE)
cat("hhh4 done:", nrow(Y), "weeks x", N, "units; fit runtime", round(runtime, 1), "s\n")
