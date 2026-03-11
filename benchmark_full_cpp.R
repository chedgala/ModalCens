# ============================================================
# Benchmark: FULL C++ Optimization vs Base R
# Strategy A: Use C++ to compute NegLogLik and let R's embedded
# C function `vmmin` (the one underlying optim BFGS) do the iterations.
# ============================================================

library(Rcpp)
library(microbenchmark)

# 1. Base R Function
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

opt_wrapper_R <- function(init_par, X, y, cens) {
  optim(par = init_par, fn = neg_loglik_R, X = X, y = y, cens = cens, 
        method = "BFGS")
}

# 2. C++ Fully Integrated Optimization (vmmin)
sourceCpp(code = '
#include <Rcpp.h>
#include <R_ext/Applic.h>

using namespace Rcpp;

// Data structure to pass to the objective function
struct OptData {
  NumericMatrix X;
  NumericVector y;
  IntegerVector cens;
  double penalty_val;
};

// Objective function evaluator
double ll_eval(int n_par, double *par_ptr, void *ex) {
  OptData *dat = (OptData *)ex;
  
  int n = dat->X.nrow();
  int p = dat->X.ncol();
  
  double phi = exp(par_ptr[p]);
  double k_ll = phi + 1.01;
  double log_k_ll = log(k_ll);
  
  double ll = 0.0;
  
  // Calculate standard negative log-likelihood for loglogistic
  for(int i = 0; i < n; ++i) {
    double eta = 0.0;
    for(int j = 0; j < p; ++j) {
      eta += dat->X(i, j) * par_ptr[j];
    }
    double moda = exp(eta);
    
    double alpha_ll = moda * pow((k_ll + 1.0) / (k_ll - 1.0), 1.0 / k_ll);
    double y_scaled = dat->y[i] / alpha_ll;
    double y_scaled_k = pow(y_scaled, k_ll);
    
    double dens = log_k_ll - log(alpha_ll) + (k_ll - 1.0) * log(y_scaled) - 2.0 * log1p(y_scaled_k);
    double prob = -log1p(y_scaled_k);
    
    ll += (1.0 - dat->cens[i]) * dens + dat->cens[i] * prob;
  }
  
  if (!R_finite(ll)) return dat->penalty_val;
  return -ll;
}

// Numerical Gradient Evaluator using central difference
void gr_eval(int n_par, double *par_ptr, double *gr_ptr, void *ex) {
  double h = 1e-7;
  for(int i = 0; i < n_par; ++i) {
    double orig = par_ptr[i];
    
    // forward
    par_ptr[i] = orig + h;
    double f1 = ll_eval(n_par, par_ptr, ex);
    
    // backward
    par_ptr[i] = orig - h;
    double f2 = ll_eval(n_par, par_ptr, ex);
    
    gr_ptr[i] = (f1 - f2) / (2.0 * h);
    
    // restore
    par_ptr[i] = orig;
  }
}

// [[Rcpp::export]]
List optim_cpp(NumericVector init_par, NumericMatrix X, NumericVector y, IntegerVector cens) {
  int n_par = init_par.size();
  
  // Create copy of parameters
  NumericVector b = clone(init_par);
  
  // Setup data struct
  OptData dat;
  dat.X = X;
  dat.y = y;
  dat.cens = cens;
  dat.penalty_val = 1e10;
  
  double Fmin = 0.0;
  int fncount = 0;
  int grcount = 0;
  int fail = 0;
  int mask[n_par];
  for(int i=0; i < n_par; ++i) mask[i] = 1;

  // Execute vmmin (BFGS native C implementation from R)
  vmmin(n_par, b.begin(), &Fmin, 
        ll_eval, gr_eval, 
        1000, 0, mask, 
        -R_PosInf, 1e-8, 10, 
        &dat, &fncount, &grcount, &fail);
        
  return List::create(
    Named("par") = b,
    Named("value") = Fmin,
    Named("counts") = IntegerVector::create(fncount, grcount),
    Named("convergence") = fail
  );
}
')

# 3. Generate Mock Data
set.seed(123)
n <- 2000  # N=2000 to see significant differences
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
C <- rexp(n, rate = 0.1)
y_obs <- pmin(y, C)
cens <- as.integer(y > C)

init_par <- c(0, 0, 0, log(1))

# 4. Check accuracy
cat("--- ACCURACY CHECK ---\n")
res_r <- opt_wrapper_R(init_par, X, y_obs, cens)
res_cpp <- optim_cpp(init_par, X, y_obs, cens)

cat("R par:", res_r$par, " val:", res_r$value, "\n")
cat("Cpp par:", res_cpp$par, " val:", res_cpp$value, "\n")

# 5. Benchmark
cat("\n--- BENCHMARK ---\n")
mb <- microbenchmark(
  Base_R   = opt_wrapper_R(init_par, X, y_obs, cens),
  Full_Cpp = optim_cpp(init_par, X, y_obs, cens),
  times = 50
)
print(mb)
