# ============================================================
# Benchmark: Rcpp Optimization vs Base R
# ============================================================

library(Rcpp)
library(microbenchmark)

# 1. Provide the pure R likelihood (from modal_reg.R)
neg_loglik_R <- function(par, X, y, cens, family = "loglogistic") {
  p    <- ncol(X)
  coef <- par[1:p]
  phi  <- exp(par[p + 1])
  
  moda <- exp(X %*% coef)
  
  if (family == "loglogistic") {
    k_ll     <- phi + 1.01
    alpha_ll <- moda * ((k_ll + 1) / (k_ll - 1))^(1 / k_ll)
    
    y_scaled <- y / alpha_ll
    dens <- log(k_ll) - log(alpha_ll) + (k_ll - 1) * log(y_scaled) - 2 * log1p(y_scaled^k_ll)
    prob <- -log1p(y_scaled^k_ll)
    
    ll <- sum((1 - cens) * dens + cens * prob)
    if (!is.finite(ll)) return(1e10)
    return(-ll)
  }
}

# 2. Provide the Rcpp implementation
cppFunction('
  double neg_loglik_cpp(NumericVector par, NumericMatrix X, NumericVector y, IntegerVector cens) {
    int n = X.nrow();
    int p = X.ncol();
    
    NumericVector coef = par[seq(0, p - 1)];
    double phi = exp(par[p]);
    
    double k_ll = phi + 1.01;
    double log_k_ll = log(k_ll);
    
    double ll = 0.0;
    
    for(int i = 0; i < n; ++i) {
      double eta = 0.0;
      for(int j = 0; j < p; ++j) {
        eta += X(i, j) * coef[j];
      }
      double moda = exp(eta);
      
      double alpha_ll = moda * pow((k_ll + 1.0) / (k_ll - 1.0), 1.0 / k_ll);
      double y_scaled = y[i] / alpha_ll;
      double y_scaled_k = pow(y_scaled, k_ll);
      
      double dens = log_k_ll - log(alpha_ll) + (k_ll - 1.0) * log(y_scaled) - 2.0 * log1p(y_scaled_k);
      double prob = -log1p(y_scaled_k);
      
      ll += (1.0 - cens[i]) * dens + cens[i] * prob;
    }
    
    if (!R_finite(ll)) return 1e10;
    return -ll;
  }
')

# 3. Wrapper for generic optim
opt_wrapper_R <- function(init_par, X, y, cens) {
  optim(par = init_par, fn = neg_loglik_R, X = X, y = y, cens = cens, 
        method = "BFGS")
}

opt_wrapper_cpp <- function(init_par, X, y, cens) {
  optim(par = init_par, fn = neg_loglik_cpp, X = X, y = y, cens = cens, 
        method = "BFGS")
}

# 4. Generate some mock data
set.seed(123)
n <- 1000  # Large N to see difference
x1 <- runif(n)
x2 <- runif(n)
X <- cbind(1, x1, x2)
gamma_true <- c(0.8, -0.3, 0.5)
phi_true <- 2.0 

mode_val <- exp(X %*% gamma_true)
k_true <- phi_true + 1.01
alpha_true <- mode_val * ((k_true + 1)/(k_true - 1))^(1/k_true)
u <- runif(n)
y <- alpha_true * (u / (1 - u))^(1/k_true)

# 15% censoring
C <- rexp(n, rate = 0.1)
y_obs <- pmin(y, C)
cens <- as.integer(y > C)

init_par <- c(0, 0, 0, log(1))

# 5. Check accuracy
cat("--- ACCURACY CHECK ---\n")
res_r <- opt_wrapper_R(init_par, X, y_obs, cens)
res_cpp <- opt_wrapper_cpp(init_par, X, y_obs, cens)

cat("R par:", res_r$par, " val:", res_r$value, "\n")
cat("Cpp par:", res_cpp$par, " val:", res_cpp$value, "\n")

# 6. Benchmark
cat("\n--- BENCHMARK ---\n")
mb <- microbenchmark(
  Base_R = opt_wrapper_R(init_par, X, y_obs, cens),
  Rcpp   = opt_wrapper_cpp(init_par, X, y_obs, cens),
  times = 50
)
print(mb)
